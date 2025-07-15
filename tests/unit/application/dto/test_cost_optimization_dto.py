"""Tests for Cost Optimization DTOs."""

from datetime import datetime

from pynomaly.application.dto.cost_optimization_dto import (
    BudgetAlertDTO,
    CloudResourceDTO,
    CostAnalysisRequestDTO,
    CostAnalysisResponseDTO,
    CostBudgetDTO,
    CostOptimizationPlanDTO,
    OptimizationPlanRequestDTO,
    OptimizationRecommendationDTO,
    RecommendationImplementationDTO,
    RecommendationImplementationResultDTO,
    ResourceCostDTO,
    ResourceSummaryDTO,
    ResourceUsageMetricsDTO,
    ServiceMetricsDTO,
)


class TestResourceUsageMetricsDTO:
    """Test suite for ResourceUsageMetricsDTO."""

    def test_valid_creation(self):
        """Test creating a valid resource usage metrics DTO."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 13, 0, 0)

        dto = ResourceUsageMetricsDTO(
            cpu_utilization_avg=65.5,
            cpu_utilization_max=89.2,
            cpu_utilization_p95=82.1,
            memory_utilization_avg=45.8,
            memory_utilization_max=67.3,
            memory_utilization_p95=58.9,
            network_bytes_in=1024000,
            network_bytes_out=512000,
            disk_read_bytes=2048000,
            disk_write_bytes=1536000,
            gpu_utilization_avg=78.4,
            storage_used_gb=128.5,
            requests_per_second=150.2,
            error_rate=0.01,
            response_time_ms=45.6,
            measurement_start=start_time,
            measurement_end=end_time,
            data_points=3600,
        )

        assert dto.cpu_utilization_avg == 65.5
        assert dto.cpu_utilization_max == 89.2
        assert dto.cpu_utilization_p95 == 82.1
        assert dto.memory_utilization_avg == 45.8
        assert dto.memory_utilization_max == 67.3
        assert dto.memory_utilization_p95 == 58.9
        assert dto.network_bytes_in == 1024000
        assert dto.network_bytes_out == 512000
        assert dto.disk_read_bytes == 2048000
        assert dto.disk_write_bytes == 1536000
        assert dto.gpu_utilization_avg == 78.4
        assert dto.storage_used_gb == 128.5
        assert dto.requests_per_second == 150.2
        assert dto.error_rate == 0.01
        assert dto.response_time_ms == 45.6
        assert dto.measurement_start == start_time
        assert dto.measurement_end == end_time
        assert dto.data_points == 3600

    def test_default_values(self):
        """Test default values."""
        dto = ResourceUsageMetricsDTO()

        assert dto.cpu_utilization_avg == 0.0
        assert dto.cpu_utilization_max == 0.0
        assert dto.cpu_utilization_p95 == 0.0
        assert dto.memory_utilization_avg == 0.0
        assert dto.memory_utilization_max == 0.0
        assert dto.memory_utilization_p95 == 0.0
        assert dto.network_bytes_in == 0
        assert dto.network_bytes_out == 0
        assert dto.disk_read_bytes == 0
        assert dto.disk_write_bytes == 0
        assert dto.gpu_utilization_avg == 0.0
        assert dto.storage_used_gb == 0.0
        assert dto.requests_per_second == 0.0
        assert dto.error_rate == 0.0
        assert dto.response_time_ms == 0.0
        assert dto.measurement_start is None
        assert dto.measurement_end is None
        assert dto.data_points == 0

    def test_high_utilization_values(self):
        """Test handling of high utilization values."""
        dto = ResourceUsageMetricsDTO(
            cpu_utilization_avg=95.5,
            cpu_utilization_max=100.0,
            cpu_utilization_p95=98.9,
            memory_utilization_avg=90.2,
            memory_utilization_max=99.8,
            memory_utilization_p95=95.1,
            gpu_utilization_avg=100.0,
            error_rate=0.15,
            response_time_ms=2000.0,
        )

        assert dto.cpu_utilization_avg == 95.5
        assert dto.cpu_utilization_max == 100.0
        assert dto.cpu_utilization_p95 == 98.9
        assert dto.memory_utilization_avg == 90.2
        assert dto.memory_utilization_max == 99.8
        assert dto.memory_utilization_p95 == 95.1
        assert dto.gpu_utilization_avg == 100.0
        assert dto.error_rate == 0.15
        assert dto.response_time_ms == 2000.0

    def test_large_data_volumes(self):
        """Test handling of large data volumes."""
        dto = ResourceUsageMetricsDTO(
            network_bytes_in=1073741824,  # 1 GB
            network_bytes_out=2147483648,  # 2 GB
            disk_read_bytes=10737418240,  # 10 GB
            disk_write_bytes=5368709120,  # 5 GB
            storage_used_gb=1024.0,  # 1 TB
            requests_per_second=10000.0,
            data_points=86400,  # 24 hours of data
        )

        assert dto.network_bytes_in == 1073741824
        assert dto.network_bytes_out == 2147483648
        assert dto.disk_read_bytes == 10737418240
        assert dto.disk_write_bytes == 5368709120
        assert dto.storage_used_gb == 1024.0
        assert dto.requests_per_second == 10000.0
        assert dto.data_points == 86400


class TestResourceCostDTO:
    """Test suite for ResourceCostDTO."""

    def test_valid_creation(self):
        """Test creating a valid resource cost DTO."""
        dto = ResourceCostDTO(
            hourly_cost=2.50,
            daily_cost=60.00,
            monthly_cost=1800.00,
            annual_cost=21600.00,
            compute_cost=1440.00,
            storage_cost=240.00,
            network_cost=120.00,
            licensing_cost=0.00,
            billing_model="reserved",
            currency="EUR",
            cost_center="engineering",
            project_id="proj-123",
            cost_trend_7d=0.05,
            cost_trend_30d=-0.02,
        )

        assert dto.hourly_cost == 2.50
        assert dto.daily_cost == 60.00
        assert dto.monthly_cost == 1800.00
        assert dto.annual_cost == 21600.00
        assert dto.compute_cost == 1440.00
        assert dto.storage_cost == 240.00
        assert dto.network_cost == 120.00
        assert dto.licensing_cost == 0.00
        assert dto.billing_model == "reserved"
        assert dto.currency == "EUR"
        assert dto.cost_center == "engineering"
        assert dto.project_id == "proj-123"
        assert dto.cost_trend_7d == 0.05
        assert dto.cost_trend_30d == -0.02

    def test_default_values(self):
        """Test default values."""
        dto = ResourceCostDTO()

        assert dto.hourly_cost == 0.0
        assert dto.daily_cost == 0.0
        assert dto.monthly_cost == 0.0
        assert dto.annual_cost == 0.0
        assert dto.compute_cost == 0.0
        assert dto.storage_cost == 0.0
        assert dto.network_cost == 0.0
        assert dto.licensing_cost == 0.0
        assert dto.billing_model == "on_demand"
        assert dto.currency == "USD"
        assert dto.cost_center is None
        assert dto.project_id is None
        assert dto.cost_trend_7d == 0.0
        assert dto.cost_trend_30d == 0.0

    def test_high_cost_values(self):
        """Test handling of high cost values."""
        dto = ResourceCostDTO(
            hourly_cost=100.00,
            daily_cost=2400.00,
            monthly_cost=72000.00,
            annual_cost=864000.00,
            compute_cost=648000.00,
            storage_cost=144000.00,
            network_cost=72000.00,
            licensing_cost=36000.00,
            cost_trend_7d=0.25,
            cost_trend_30d=0.15,
        )

        assert dto.hourly_cost == 100.00
        assert dto.daily_cost == 2400.00
        assert dto.monthly_cost == 72000.00
        assert dto.annual_cost == 864000.00
        assert dto.compute_cost == 648000.00
        assert dto.storage_cost == 144000.00
        assert dto.network_cost == 72000.00
        assert dto.licensing_cost == 36000.00
        assert dto.cost_trend_7d == 0.25
        assert dto.cost_trend_30d == 0.15

    def test_different_currencies(self):
        """Test different currency values."""
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]

        for currency in currencies:
            dto = ResourceCostDTO(hourly_cost=5.00, currency=currency)
            assert dto.currency == currency

    def test_billing_models(self):
        """Test different billing models."""
        billing_models = ["on_demand", "reserved", "spot", "dedicated"]

        for model in billing_models:
            dto = ResourceCostDTO(hourly_cost=3.00, billing_model=model)
            assert dto.billing_model == model


class TestCloudResourceDTO:
    """Test suite for CloudResourceDTO."""

    def test_valid_creation(self):
        """Test creating a valid cloud resource DTO."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        last_accessed = datetime(2023, 1, 15, 14, 30, 0)
        scheduled_termination = datetime(2023, 2, 1, 0, 0, 0)

        usage_metrics = ResourceUsageMetricsDTO(
            cpu_utilization_avg=45.0, memory_utilization_avg=65.0
        )

        cost_info = ResourceCostDTO(hourly_cost=1.50, daily_cost=36.00)

        dto = CloudResourceDTO(
            resource_id="r-1234567890abcdef0",
            name="web-server-01",
            resource_type="ec2",
            provider="aws",
            region="us-east-1",
            availability_zone="us-east-1a",
            instance_type="t3.medium",
            cpu_cores=2,
            memory_gb=4.0,
            storage_gb=20.0,
            gpu_count=0,
            network_performance="up_to_5_gbps",
            status="running",
            created_at=created_at,
            last_accessed=last_accessed,
            scheduled_termination=scheduled_termination,
            usage_metrics=usage_metrics,
            cost_info=cost_info,
            owner="john.doe@company.com",
            team="backend",
            environment="staging",
            tenant_id="tenant-123",
            tags={"project": "web-app", "environment": "staging"},
        )

        assert dto.resource_id == "r-1234567890abcdef0"
        assert dto.name == "web-server-01"
        assert dto.resource_type == "ec2"
        assert dto.provider == "aws"
        assert dto.region == "us-east-1"
        assert dto.availability_zone == "us-east-1a"
        assert dto.instance_type == "t3.medium"
        assert dto.cpu_cores == 2
        assert dto.memory_gb == 4.0
        assert dto.storage_gb == 20.0
        assert dto.gpu_count == 0
        assert dto.network_performance == "up_to_5_gbps"
        assert dto.status == "running"
        assert dto.created_at == created_at
        assert dto.last_accessed == last_accessed
        assert dto.scheduled_termination == scheduled_termination
        assert dto.usage_metrics == usage_metrics
        assert dto.cost_info == cost_info
        assert dto.owner == "john.doe@company.com"
        assert dto.team == "backend"
        assert dto.environment == "staging"
        assert dto.tenant_id == "tenant-123"
        assert dto.tags == {"project": "web-app", "environment": "staging"}

    def test_default_values(self):
        """Test default values."""
        dto = CloudResourceDTO()

        assert dto.resource_id is None
        assert dto.name == ""
        assert dto.resource_type == "compute"
        assert dto.provider == "aws"
        assert dto.region == ""
        assert dto.availability_zone == ""
        assert dto.instance_type == ""
        assert dto.cpu_cores == 0
        assert dto.memory_gb == 0.0
        assert dto.storage_gb == 0.0
        assert dto.gpu_count == 0
        assert dto.network_performance == ""
        assert dto.status == "active"
        assert dto.created_at is None
        assert dto.last_accessed is None
        assert dto.scheduled_termination is None
        assert dto.usage_metrics is None
        assert dto.cost_info is None
        assert dto.owner is None
        assert dto.team is None
        assert dto.environment == "production"
        assert dto.tenant_id is None
        assert dto.tags == {}

    def test_post_init_tags_initialization(self):
        """Test that __post_init__ properly initializes tags."""
        dto = CloudResourceDTO(name="test-resource", tags=None)
        assert dto.tags == {}

    def test_different_providers(self):
        """Test different cloud providers."""
        providers = ["aws", "azure", "gcp", "alibaba"]

        for provider in providers:
            dto = CloudResourceDTO(name=f"resource-{provider}", provider=provider)
            assert dto.provider == provider

    def test_different_resource_types(self):
        """Test different resource types."""
        resource_types = ["compute", "storage", "network", "database", "cache"]

        for resource_type in resource_types:
            dto = CloudResourceDTO(
                name=f"resource-{resource_type}", resource_type=resource_type
            )
            assert dto.resource_type == resource_type

    def test_different_environments(self):
        """Test different environments."""
        environments = ["development", "staging", "production", "testing"]

        for environment in environments:
            dto = CloudResourceDTO(
                name=f"resource-{environment}", environment=environment
            )
            assert dto.environment == environment

    def test_different_statuses(self):
        """Test different resource statuses."""
        statuses = ["active", "inactive", "running", "stopped", "terminated"]

        for status in statuses:
            dto = CloudResourceDTO(name=f"resource-{status}", status=status)
            assert dto.status == status

    def test_gpu_resources(self):
        """Test GPU-enabled resources."""
        dto = CloudResourceDTO(
            name="gpu-instance",
            instance_type="p3.2xlarge",
            cpu_cores=8,
            memory_gb=61.0,
            gpu_count=1,
            network_performance="up_to_10_gbps",
        )

        assert dto.instance_type == "p3.2xlarge"
        assert dto.cpu_cores == 8
        assert dto.memory_gb == 61.0
        assert dto.gpu_count == 1
        assert dto.network_performance == "up_to_10_gbps"

    def test_high_capacity_resources(self):
        """Test high-capacity resources."""
        dto = CloudResourceDTO(
            name="high-capacity-instance",
            instance_type="r5.24xlarge",
            cpu_cores=96,
            memory_gb=768.0,
            storage_gb=2000.0,
            gpu_count=4,
        )

        assert dto.cpu_cores == 96
        assert dto.memory_gb == 768.0
        assert dto.storage_gb == 2000.0
        assert dto.gpu_count == 4


class TestOptimizationRecommendationDTO:
    """Test suite for OptimizationRecommendationDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization recommendation DTO."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        expires_at = datetime(2023, 1, 31, 23, 59, 59)

        dto = OptimizationRecommendationDTO(
            recommendation_id="rec-123",
            resource_id="r-1234567890abcdef0",
            recommendation_type="instance_resize",
            priority="high",
            title="Downsize underutilized instance",
            description="Instance is consistently underutilized",
            rationale="CPU utilization is below 20% for 30 days",
            current_monthly_cost=720.00,
            projected_monthly_cost=360.00,
            monthly_savings=360.00,
            annual_savings=4320.00,
            implementation_cost=0.00,
            payback_period_days=0,
            performance_impact="minimal",
            risk_level="low",
            confidence_score=0.95,
            action_required="resize_instance",
            automation_possible=True,
            estimated_implementation_time="5 minutes",
            prerequisites=["stop_instance", "backup_data"],
            created_at=created_at,
            created_by="cost_optimizer_v2",
            status="approved",
            expires_at=expires_at,
            affected_resources=["r-1234567890abcdef0"],
            dependencies=["dep-456"],
        )

        assert dto.recommendation_id == "rec-123"
        assert dto.resource_id == "r-1234567890abcdef0"
        assert dto.recommendation_type == "instance_resize"
        assert dto.priority == "high"
        assert dto.title == "Downsize underutilized instance"
        assert dto.description == "Instance is consistently underutilized"
        assert dto.rationale == "CPU utilization is below 20% for 30 days"
        assert dto.current_monthly_cost == 720.00
        assert dto.projected_monthly_cost == 360.00
        assert dto.monthly_savings == 360.00
        assert dto.annual_savings == 4320.00
        assert dto.implementation_cost == 0.00
        assert dto.payback_period_days == 0
        assert dto.performance_impact == "minimal"
        assert dto.risk_level == "low"
        assert dto.confidence_score == 0.95
        assert dto.action_required == "resize_instance"
        assert dto.automation_possible is True
        assert dto.estimated_implementation_time == "5 minutes"
        assert dto.prerequisites == ["stop_instance", "backup_data"]
        assert dto.created_at == created_at
        assert dto.created_by == "cost_optimizer_v2"
        assert dto.status == "approved"
        assert dto.expires_at == expires_at
        assert dto.affected_resources == ["r-1234567890abcdef0"]
        assert dto.dependencies == ["dep-456"]

    def test_default_values(self):
        """Test default values."""
        dto = OptimizationRecommendationDTO()

        assert dto.recommendation_id is None
        assert dto.resource_id is None
        assert dto.recommendation_type == "rightsizing"
        assert dto.priority == "medium"
        assert dto.title == ""
        assert dto.description == ""
        assert dto.rationale == ""
        assert dto.current_monthly_cost == 0.0
        assert dto.projected_monthly_cost == 0.0
        assert dto.monthly_savings == 0.0
        assert dto.annual_savings == 0.0
        assert dto.implementation_cost == 0.0
        assert dto.payback_period_days == 0
        assert dto.performance_impact == "none"
        assert dto.risk_level == "low"
        assert dto.confidence_score == 0.8
        assert dto.action_required == ""
        assert dto.automation_possible is False
        assert dto.estimated_implementation_time == ""
        assert dto.prerequisites == []
        assert dto.created_at is None
        assert dto.created_by == "cost_optimizer"
        assert dto.status == "pending"
        assert dto.expires_at is None
        assert dto.affected_resources == []
        assert dto.dependencies == []

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = OptimizationRecommendationDTO(
            prerequisites=None, affected_resources=None, dependencies=None
        )

        assert dto.prerequisites == []
        assert dto.affected_resources == []
        assert dto.dependencies == []

    def test_different_recommendation_types(self):
        """Test different recommendation types."""
        recommendation_types = [
            "rightsizing",
            "instance_resize",
            "scheduling",
            "idle_cleanup",
            "storage_optimization",
            "reserved_instances",
            "spot_instances",
        ]

        for rec_type in recommendation_types:
            dto = OptimizationRecommendationDTO(recommendation_type=rec_type)
            assert dto.recommendation_type == rec_type

    def test_different_priorities(self):
        """Test different priority levels."""
        priorities = ["low", "medium", "high", "critical"]

        for priority in priorities:
            dto = OptimizationRecommendationDTO(priority=priority)
            assert dto.priority == priority

    def test_different_risk_levels(self):
        """Test different risk levels."""
        risk_levels = ["low", "medium", "high", "critical"]

        for risk_level in risk_levels:
            dto = OptimizationRecommendationDTO(risk_level=risk_level)
            assert dto.risk_level == risk_level

    def test_different_performance_impacts(self):
        """Test different performance impact levels."""
        performance_impacts = ["none", "minimal", "moderate", "significant"]

        for impact in performance_impacts:
            dto = OptimizationRecommendationDTO(performance_impact=impact)
            assert dto.performance_impact == impact

    def test_different_statuses(self):
        """Test different recommendation statuses."""
        statuses = ["pending", "approved", "rejected", "implemented", "expired"]

        for status in statuses:
            dto = OptimizationRecommendationDTO(status=status)
            assert dto.status == status

    def test_high_savings_recommendation(self):
        """Test recommendation with high savings."""
        dto = OptimizationRecommendationDTO(
            recommendation_type="instance_resize",
            priority="high",
            current_monthly_cost=5000.00,
            projected_monthly_cost=1000.00,
            monthly_savings=4000.00,
            annual_savings=48000.00,
            confidence_score=0.98,
        )

        assert dto.current_monthly_cost == 5000.00
        assert dto.projected_monthly_cost == 1000.00
        assert dto.monthly_savings == 4000.00
        assert dto.annual_savings == 48000.00
        assert dto.confidence_score == 0.98

    def test_complex_prerequisites(self):
        """Test recommendation with complex prerequisites."""
        dto = OptimizationRecommendationDTO(
            prerequisites=[
                "backup_data",
                "notify_stakeholders",
                "schedule_maintenance_window",
                "update_monitoring_alerts",
                "test_failover_procedures",
            ],
            affected_resources=[
                "r-1234567890abcdef0",
                "r-abcdef1234567890",
                "r-567890abcdef1234",
            ],
            dependencies=[
                "dep-database-migration",
                "dep-load-balancer-update",
                "dep-dns-update",
            ],
        )

        assert len(dto.prerequisites) == 5
        assert len(dto.affected_resources) == 3
        assert len(dto.dependencies) == 3
        assert "backup_data" in dto.prerequisites
        assert "r-1234567890abcdef0" in dto.affected_resources
        assert "dep-database-migration" in dto.dependencies


class TestCostBudgetDTO:
    """Test suite for CostBudgetDTO."""

    def test_valid_creation(self):
        """Test creating a valid cost budget DTO."""
        created_at = datetime(2023, 1, 1, 0, 0, 0)
        last_updated = datetime(2023, 1, 15, 10, 30, 0)

        dto = CostBudgetDTO(
            budget_id="budget-123",
            name="Engineering Team Budget",
            description="Monthly budget for engineering resources",
            monthly_limit=10000.00,
            annual_limit=120000.00,
            currency="USD",
            tenant_id="tenant-123",
            resource_types=["compute", "storage", "network"],
            environments=["development", "staging", "production"],
            cost_centers=["engineering", "devops"],
            current_monthly_spend=7500.00,
            current_annual_spend=90000.00,
            projected_monthly_spend=9200.00,
            projected_annual_spend=110400.00,
            alert_thresholds=[0.6, 0.8, 0.9, 1.0],
            alert_contacts=["team-lead@company.com", "finance@company.com"],
            auto_actions_enabled=True,
            created_at=created_at,
            created_by="budget.admin",
            last_updated=last_updated,
        )

        assert dto.budget_id == "budget-123"
        assert dto.name == "Engineering Team Budget"
        assert dto.description == "Monthly budget for engineering resources"
        assert dto.monthly_limit == 10000.00
        assert dto.annual_limit == 120000.00
        assert dto.currency == "USD"
        assert dto.tenant_id == "tenant-123"
        assert dto.resource_types == ["compute", "storage", "network"]
        assert dto.environments == ["development", "staging", "production"]
        assert dto.cost_centers == ["engineering", "devops"]
        assert dto.current_monthly_spend == 7500.00
        assert dto.current_annual_spend == 90000.00
        assert dto.projected_monthly_spend == 9200.00
        assert dto.projected_annual_spend == 110400.00
        assert dto.alert_thresholds == [0.6, 0.8, 0.9, 1.0]
        assert dto.alert_contacts == ["team-lead@company.com", "finance@company.com"]
        assert dto.auto_actions_enabled is True
        assert dto.created_at == created_at
        assert dto.created_by == "budget.admin"
        assert dto.last_updated == last_updated

    def test_default_values(self):
        """Test default values."""
        dto = CostBudgetDTO()

        assert dto.budget_id is None
        assert dto.name == ""
        assert dto.description == ""
        assert dto.monthly_limit == 0.0
        assert dto.annual_limit == 0.0
        assert dto.currency == "USD"
        assert dto.tenant_id is None
        assert dto.resource_types == []
        assert dto.environments == []
        assert dto.cost_centers == []
        assert dto.current_monthly_spend == 0.0
        assert dto.current_annual_spend == 0.0
        assert dto.projected_monthly_spend == 0.0
        assert dto.projected_annual_spend == 0.0
        assert dto.alert_thresholds == [0.5, 0.8, 0.9, 1.0]
        assert dto.alert_contacts == []
        assert dto.auto_actions_enabled is False
        assert dto.created_at is None
        assert dto.created_by == ""
        assert dto.last_updated is None

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = CostBudgetDTO(
            resource_types=None,
            environments=None,
            cost_centers=None,
            alert_thresholds=None,
            alert_contacts=None,
        )

        assert dto.resource_types == []
        assert dto.environments == []
        assert dto.cost_centers == []
        assert dto.alert_thresholds == [0.5, 0.8, 0.9, 1.0]
        assert dto.alert_contacts == []

    def test_different_currencies(self):
        """Test different currencies."""
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]

        for currency in currencies:
            dto = CostBudgetDTO(name=f"Budget {currency}", currency=currency)
            assert dto.currency == currency

    def test_large_budget_values(self):
        """Test handling of large budget values."""
        dto = CostBudgetDTO(
            name="Enterprise Budget",
            monthly_limit=1000000.00,
            annual_limit=12000000.00,
            current_monthly_spend=800000.00,
            current_annual_spend=9600000.00,
            projected_monthly_spend=950000.00,
            projected_annual_spend=11400000.00,
        )

        assert dto.monthly_limit == 1000000.00
        assert dto.annual_limit == 12000000.00
        assert dto.current_monthly_spend == 800000.00
        assert dto.current_annual_spend == 9600000.00
        assert dto.projected_monthly_spend == 950000.00
        assert dto.projected_annual_spend == 11400000.00

    def test_custom_alert_thresholds(self):
        """Test custom alert thresholds."""
        dto = CostBudgetDTO(
            name="Custom Alerts Budget",
            alert_thresholds=[0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
        )

        assert dto.alert_thresholds == [0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

    def test_multiple_resource_types(self):
        """Test multiple resource types."""
        dto = CostBudgetDTO(
            name="Multi-Resource Budget",
            resource_types=[
                "compute",
                "storage",
                "network",
                "database",
                "cache",
                "cdn",
            ],
        )

        assert len(dto.resource_types) == 6
        assert "compute" in dto.resource_types
        assert "storage" in dto.resource_types
        assert "network" in dto.resource_types
        assert "database" in dto.resource_types
        assert "cache" in dto.resource_types
        assert "cdn" in dto.resource_types

    def test_multiple_environments(self):
        """Test multiple environments."""
        dto = CostBudgetDTO(
            name="Multi-Environment Budget",
            environments=["development", "testing", "staging", "production", "sandbox"],
        )

        assert len(dto.environments) == 5
        assert "development" in dto.environments
        assert "testing" in dto.environments
        assert "staging" in dto.environments
        assert "production" in dto.environments
        assert "sandbox" in dto.environments

    def test_multiple_cost_centers(self):
        """Test multiple cost centers."""
        dto = CostBudgetDTO(
            name="Multi-Cost-Center Budget",
            cost_centers=["engineering", "devops", "data-science", "qa", "security"],
        )

        assert len(dto.cost_centers) == 5
        assert "engineering" in dto.cost_centers
        assert "devops" in dto.cost_centers
        assert "data-science" in dto.cost_centers
        assert "qa" in dto.cost_centers
        assert "security" in dto.cost_centers

    def test_multiple_alert_contacts(self):
        """Test multiple alert contacts."""
        dto = CostBudgetDTO(
            name="Multi-Contact Budget",
            alert_contacts=[
                "team-lead@company.com",
                "finance@company.com",
                "cto@company.com",
                "alerts@company.com",
            ],
        )

        assert len(dto.alert_contacts) == 4
        assert "team-lead@company.com" in dto.alert_contacts
        assert "finance@company.com" in dto.alert_contacts
        assert "cto@company.com" in dto.alert_contacts
        assert "alerts@company.com" in dto.alert_contacts


class TestCostOptimizationPlanDTO:
    """Test suite for CostOptimizationPlanDTO."""

    def test_valid_creation(self):
        """Test creating a valid cost optimization plan DTO."""
        created_at = datetime(2023, 1, 1, 0, 0, 0)

        recommendations = [
            OptimizationRecommendationDTO(
                recommendation_type="rightsizing", monthly_savings=500.00
            ),
            OptimizationRecommendationDTO(
                recommendation_type="scheduling", monthly_savings=300.00
            ),
        ]

        dto = CostOptimizationPlanDTO(
            plan_id="plan-123",
            name="Q1 2023 Optimization Plan",
            description="Comprehensive cost optimization for Q1",
            strategy="aggressive",
            tenant_id="tenant-123",
            resource_scope=["compute", "storage"],
            environments=["production", "staging"],
            target_cost_reduction_percent=25.0,
            target_monthly_savings=5000.00,
            max_performance_impact="moderate",
            max_risk_level="high",
            recommendations=recommendations,
            total_potential_savings=800.00,
            total_implementation_cost=100.00,
            estimated_implementation_days=5,
            approved_recommendations=2,
            implemented_recommendations=1,
            actual_savings_to_date=500.00,
            created_at=created_at,
            created_by="cost.manager",
            status="in_progress",
        )

        assert dto.plan_id == "plan-123"
        assert dto.name == "Q1 2023 Optimization Plan"
        assert dto.description == "Comprehensive cost optimization for Q1"
        assert dto.strategy == "aggressive"
        assert dto.tenant_id == "tenant-123"
        assert dto.resource_scope == ["compute", "storage"]
        assert dto.environments == ["production", "staging"]
        assert dto.target_cost_reduction_percent == 25.0
        assert dto.target_monthly_savings == 5000.00
        assert dto.max_performance_impact == "moderate"
        assert dto.max_risk_level == "high"
        assert len(dto.recommendations) == 2
        assert dto.total_potential_savings == 800.00
        assert dto.total_implementation_cost == 100.00
        assert dto.estimated_implementation_days == 5
        assert dto.approved_recommendations == 2
        assert dto.implemented_recommendations == 1
        assert dto.actual_savings_to_date == 500.00
        assert dto.created_at == created_at
        assert dto.created_by == "cost.manager"
        assert dto.status == "in_progress"

    def test_default_values(self):
        """Test default values."""
        dto = CostOptimizationPlanDTO()

        assert dto.plan_id is None
        assert dto.name == ""
        assert dto.description == ""
        assert dto.strategy == "balanced"
        assert dto.tenant_id is None
        assert dto.resource_scope == []
        assert dto.environments == []
        assert dto.target_cost_reduction_percent == 0.0
        assert dto.target_monthly_savings == 0.0
        assert dto.max_performance_impact == "minimal"
        assert dto.max_risk_level == "medium"
        assert dto.recommendations == []
        assert dto.total_potential_savings == 0.0
        assert dto.total_implementation_cost == 0.0
        assert dto.estimated_implementation_days == 0
        assert dto.approved_recommendations == 0
        assert dto.implemented_recommendations == 0
        assert dto.actual_savings_to_date == 0.0
        assert dto.created_at is None
        assert dto.created_by == ""
        assert dto.status == "draft"

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = CostOptimizationPlanDTO(
            resource_scope=None, environments=None, recommendations=None
        )

        assert dto.resource_scope == []
        assert dto.environments == []
        assert dto.recommendations == []

    def test_different_strategies(self):
        """Test different optimization strategies."""
        strategies = ["conservative", "balanced", "aggressive", "experimental"]

        for strategy in strategies:
            dto = CostOptimizationPlanDTO(name=f"Plan {strategy}", strategy=strategy)
            assert dto.strategy == strategy

    def test_different_performance_impacts(self):
        """Test different performance impact levels."""
        performance_impacts = ["none", "minimal", "moderate", "significant"]

        for impact in performance_impacts:
            dto = CostOptimizationPlanDTO(
                name=f"Plan {impact}", max_performance_impact=impact
            )
            assert dto.max_performance_impact == impact

    def test_different_risk_levels(self):
        """Test different risk levels."""
        risk_levels = ["low", "medium", "high", "critical"]

        for risk_level in risk_levels:
            dto = CostOptimizationPlanDTO(
                name=f"Plan {risk_level}", max_risk_level=risk_level
            )
            assert dto.max_risk_level == risk_level

    def test_different_statuses(self):
        """Test different plan statuses."""
        statuses = ["draft", "approved", "in_progress", "completed", "cancelled"]

        for status in statuses:
            dto = CostOptimizationPlanDTO(name=f"Plan {status}", status=status)
            assert dto.status == status

    def test_high_savings_targets(self):
        """Test high savings targets."""
        dto = CostOptimizationPlanDTO(
            name="High Savings Plan",
            target_cost_reduction_percent=40.0,
            target_monthly_savings=50000.00,
            total_potential_savings=60000.00,
        )

        assert dto.target_cost_reduction_percent == 40.0
        assert dto.target_monthly_savings == 50000.00
        assert dto.total_potential_savings == 60000.00

    def test_complex_resource_scope(self):
        """Test complex resource scope."""
        dto = CostOptimizationPlanDTO(
            name="Complex Scope Plan",
            resource_scope=[
                "compute",
                "storage",
                "network",
                "database",
                "cache",
                "cdn",
                "monitoring",
                "logging",
            ],
        )

        assert len(dto.resource_scope) == 8
        assert "compute" in dto.resource_scope
        assert "storage" in dto.resource_scope
        assert "network" in dto.resource_scope
        assert "database" in dto.resource_scope
        assert "cache" in dto.resource_scope
        assert "cdn" in dto.resource_scope
        assert "monitoring" in dto.resource_scope
        assert "logging" in dto.resource_scope

    def test_multiple_environments(self):
        """Test multiple environments."""
        dto = CostOptimizationPlanDTO(
            name="Multi-Environment Plan",
            environments=["development", "testing", "staging", "production", "sandbox"],
        )

        assert len(dto.environments) == 5
        assert "development" in dto.environments
        assert "testing" in dto.environments
        assert "staging" in dto.environments
        assert "production" in dto.environments
        assert "sandbox" in dto.environments

    def test_plan_with_many_recommendations(self):
        """Test plan with many recommendations."""
        recommendations = [
            OptimizationRecommendationDTO(
                recommendation_type="rightsizing", monthly_savings=200.00
            )
            for _ in range(10)
        ]

        dto = CostOptimizationPlanDTO(
            name="Large Recommendation Plan",
            recommendations=recommendations,
            total_potential_savings=2000.00,
            approved_recommendations=8,
            implemented_recommendations=5,
        )

        assert len(dto.recommendations) == 10
        assert dto.total_potential_savings == 2000.00
        assert dto.approved_recommendations == 8
        assert dto.implemented_recommendations == 5

    def test_implementation_metrics(self):
        """Test implementation metrics."""
        dto = CostOptimizationPlanDTO(
            name="Implementation Metrics Plan",
            estimated_implementation_days=30,
            approved_recommendations=15,
            implemented_recommendations=10,
            actual_savings_to_date=7500.00,
        )

        assert dto.estimated_implementation_days == 30
        assert dto.approved_recommendations == 15
        assert dto.implemented_recommendations == 10
        assert dto.actual_savings_to_date == 7500.00


class TestCostAnalysisRequestDTO:
    """Test suite for CostAnalysisRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid cost analysis request DTO."""
        dto = CostAnalysisRequestDTO(
            tenant_id="tenant-123",
            days=60,
            include_predictions=True,
            include_anomalies=True,
            include_optimization_potential=True,
            resource_types=["compute", "storage", "network"],
            environments=["production", "staging"],
            providers=["aws", "azure"],
        )

        assert dto.tenant_id == "tenant-123"
        assert dto.days == 60
        assert dto.include_predictions is True
        assert dto.include_anomalies is True
        assert dto.include_optimization_potential is True
        assert dto.resource_types == ["compute", "storage", "network"]
        assert dto.environments == ["production", "staging"]
        assert dto.providers == ["aws", "azure"]

    def test_default_values(self):
        """Test default values."""
        dto = CostAnalysisRequestDTO()

        assert dto.tenant_id is None
        assert dto.days == 30
        assert dto.include_predictions is True
        assert dto.include_anomalies is True
        assert dto.include_optimization_potential is True
        assert dto.resource_types == []
        assert dto.environments == []
        assert dto.providers == []

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = CostAnalysisRequestDTO(
            resource_types=None, environments=None, providers=None
        )

        assert dto.resource_types == []
        assert dto.environments == []
        assert dto.providers == []

    def test_different_day_ranges(self):
        """Test different day ranges."""
        day_ranges = [7, 14, 30, 60, 90, 180, 365]

        for days in day_ranges:
            dto = CostAnalysisRequestDTO(days=days)
            assert dto.days == days

    def test_selective_analysis_options(self):
        """Test selective analysis options."""
        dto = CostAnalysisRequestDTO(
            include_predictions=False,
            include_anomalies=True,
            include_optimization_potential=False,
        )

        assert dto.include_predictions is False
        assert dto.include_anomalies is True
        assert dto.include_optimization_potential is False

    def test_multiple_resource_types(self):
        """Test multiple resource types."""
        dto = CostAnalysisRequestDTO(
            resource_types=["compute", "storage", "network", "database", "cache", "cdn"]
        )

        assert len(dto.resource_types) == 6
        assert "compute" in dto.resource_types
        assert "storage" in dto.resource_types
        assert "network" in dto.resource_types
        assert "database" in dto.resource_types
        assert "cache" in dto.resource_types
        assert "cdn" in dto.resource_types

    def test_multiple_environments(self):
        """Test multiple environments."""
        dto = CostAnalysisRequestDTO(
            environments=["development", "testing", "staging", "production", "sandbox"]
        )

        assert len(dto.environments) == 5
        assert "development" in dto.environments
        assert "testing" in dto.environments
        assert "staging" in dto.environments
        assert "production" in dto.environments
        assert "sandbox" in dto.environments

    def test_multiple_providers(self):
        """Test multiple providers."""
        dto = CostAnalysisRequestDTO(
            providers=["aws", "azure", "gcp", "alibaba", "oracle"]
        )

        assert len(dto.providers) == 5
        assert "aws" in dto.providers
        assert "azure" in dto.providers
        assert "gcp" in dto.providers
        assert "alibaba" in dto.providers
        assert "oracle" in dto.providers


class TestCostAnalysisResponseDTO:
    """Test suite for CostAnalysisResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid cost analysis response DTO."""
        generated_at = datetime(2023, 1, 15, 10, 30, 0)

        dto = CostAnalysisResponseDTO(
            analysis_id="analysis-123",
            generated_at=generated_at,
            total_monthly_cost=25000.00,
            projected_annual_cost=300000.00,
            cost_by_resource_type={
                "compute": 15000.00,
                "storage": 8000.00,
                "network": 2000.00,
            },
            cost_by_provider={"aws": 20000.00, "azure": 5000.00},
            cost_by_environment={
                "production": 18000.00,
                "staging": 5000.00,
                "development": 2000.00,
            },
            top_cost_drivers=[
                {"resource_id": "r-123", "cost": 5000.00, "type": "compute"},
                {"resource_id": "r-456", "cost": 3000.00, "type": "storage"},
            ],
            cost_trends={"7d": 0.05, "30d": 0.12, "90d": 0.08},
            inefficiency_indicators={
                "idle_resources": 15,
                "underutilized_resources": 8,
            },
            cost_anomalies=[
                {
                    "date": "2023-01-14",
                    "cost": 5000.00,
                    "expected": 2500.00,
                    "severity": "high",
                }
            ],
            cost_predictions={"next_month": 26000.00, "next_quarter": 78000.00},
            optimization_potential={"monthly_savings": 3000.00, "recommendations": 12},
        )

        assert dto.analysis_id == "analysis-123"
        assert dto.generated_at == generated_at
        assert dto.total_monthly_cost == 25000.00
        assert dto.projected_annual_cost == 300000.00
        assert dto.cost_by_resource_type == {
            "compute": 15000.00,
            "storage": 8000.00,
            "network": 2000.00,
        }
        assert dto.cost_by_provider == {"aws": 20000.00, "azure": 5000.00}
        assert dto.cost_by_environment == {
            "production": 18000.00,
            "staging": 5000.00,
            "development": 2000.00,
        }
        assert len(dto.top_cost_drivers) == 2
        assert dto.cost_trends == {"7d": 0.05, "30d": 0.12, "90d": 0.08}
        assert dto.inefficiency_indicators == {
            "idle_resources": 15,
            "underutilized_resources": 8,
        }
        assert len(dto.cost_anomalies) == 1
        assert dto.cost_predictions == {
            "next_month": 26000.00,
            "next_quarter": 78000.00,
        }
        assert dto.optimization_potential == {
            "monthly_savings": 3000.00,
            "recommendations": 12,
        }

    def test_default_values(self):
        """Test default values."""
        dto = CostAnalysisResponseDTO()

        assert dto.analysis_id == ""
        assert dto.generated_at is None
        assert dto.total_monthly_cost == 0.0
        assert dto.projected_annual_cost == 0.0
        assert dto.cost_by_resource_type == {}
        assert dto.cost_by_provider == {}
        assert dto.cost_by_environment == {}
        assert dto.top_cost_drivers == []
        assert dto.cost_trends == {}
        assert dto.inefficiency_indicators == {}
        assert dto.cost_anomalies == []
        assert dto.cost_predictions == {}
        assert dto.optimization_potential == {}

    def test_post_init_dict_and_list_initialization(self):
        """Test that __post_init__ properly initializes dicts and lists."""
        dto = CostAnalysisResponseDTO(
            cost_by_resource_type=None,
            cost_by_provider=None,
            cost_by_environment=None,
            top_cost_drivers=None,
            cost_trends=None,
            inefficiency_indicators=None,
            cost_anomalies=None,
            cost_predictions=None,
            optimization_potential=None,
        )

        assert dto.cost_by_resource_type == {}
        assert dto.cost_by_provider == {}
        assert dto.cost_by_environment == {}
        assert dto.top_cost_drivers == []
        assert dto.cost_trends == {}
        assert dto.inefficiency_indicators == {}
        assert dto.cost_anomalies == []
        assert dto.cost_predictions == {}
        assert dto.optimization_potential == {}

    def test_high_cost_values(self):
        """Test handling of high cost values."""
        dto = CostAnalysisResponseDTO(
            total_monthly_cost=1000000.00,
            projected_annual_cost=12000000.00,
            cost_by_resource_type={
                "compute": 600000.00,
                "storage": 300000.00,
                "network": 100000.00,
            },
        )

        assert dto.total_monthly_cost == 1000000.00
        assert dto.projected_annual_cost == 12000000.00
        assert dto.cost_by_resource_type["compute"] == 600000.00
        assert dto.cost_by_resource_type["storage"] == 300000.00
        assert dto.cost_by_resource_type["network"] == 100000.00

    def test_multiple_providers(self):
        """Test multiple providers in cost breakdown."""
        dto = CostAnalysisResponseDTO(
            cost_by_provider={
                "aws": 15000.00,
                "azure": 8000.00,
                "gcp": 5000.00,
                "alibaba": 2000.00,
            }
        )

        assert dto.cost_by_provider["aws"] == 15000.00
        assert dto.cost_by_provider["azure"] == 8000.00
        assert dto.cost_by_provider["gcp"] == 5000.00
        assert dto.cost_by_provider["alibaba"] == 2000.00

    def test_multiple_environments(self):
        """Test multiple environments in cost breakdown."""
        dto = CostAnalysisResponseDTO(
            cost_by_environment={
                "production": 20000.00,
                "staging": 6000.00,
                "development": 3000.00,
                "testing": 1000.00,
                "sandbox": 500.00,
            }
        )

        assert dto.cost_by_environment["production"] == 20000.00
        assert dto.cost_by_environment["staging"] == 6000.00
        assert dto.cost_by_environment["development"] == 3000.00
        assert dto.cost_by_environment["testing"] == 1000.00
        assert dto.cost_by_environment["sandbox"] == 500.00

    def test_many_top_cost_drivers(self):
        """Test many top cost drivers."""
        cost_drivers = [
            {"resource_id": f"r-{i}", "cost": 1000.00 - (i * 100), "type": "compute"}
            for i in range(10)
        ]

        dto = CostAnalysisResponseDTO(top_cost_drivers=cost_drivers)

        assert len(dto.top_cost_drivers) == 10
        assert dto.top_cost_drivers[0]["cost"] == 1000.00
        assert dto.top_cost_drivers[9]["cost"] == 100.00

    def test_complex_cost_trends(self):
        """Test complex cost trends."""
        dto = CostAnalysisResponseDTO(
            cost_trends={
                "1d": 0.02,
                "7d": 0.05,
                "14d": 0.08,
                "30d": 0.12,
                "60d": 0.15,
                "90d": 0.18,
                "180d": 0.25,
                "365d": 0.30,
            }
        )

        assert dto.cost_trends["1d"] == 0.02
        assert dto.cost_trends["7d"] == 0.05
        assert dto.cost_trends["14d"] == 0.08
        assert dto.cost_trends["30d"] == 0.12
        assert dto.cost_trends["60d"] == 0.15
        assert dto.cost_trends["90d"] == 0.18
        assert dto.cost_trends["180d"] == 0.25
        assert dto.cost_trends["365d"] == 0.30

    def test_detailed_inefficiency_indicators(self):
        """Test detailed inefficiency indicators."""
        dto = CostAnalysisResponseDTO(
            inefficiency_indicators={
                "idle_resources": 25,
                "underutilized_resources": 18,
                "oversized_resources": 12,
                "untagged_resources": 8,
                "duplicate_resources": 3,
                "unused_storage": 15,
                "idle_load_balancers": 5,
                "unused_ip_addresses": 10,
            }
        )

        assert dto.inefficiency_indicators["idle_resources"] == 25
        assert dto.inefficiency_indicators["underutilized_resources"] == 18
        assert dto.inefficiency_indicators["oversized_resources"] == 12
        assert dto.inefficiency_indicators["untagged_resources"] == 8
        assert dto.inefficiency_indicators["duplicate_resources"] == 3
        assert dto.inefficiency_indicators["unused_storage"] == 15
        assert dto.inefficiency_indicators["idle_load_balancers"] == 5
        assert dto.inefficiency_indicators["unused_ip_addresses"] == 10

    def test_multiple_cost_anomalies(self):
        """Test multiple cost anomalies."""
        anomalies = [
            {
                "date": "2023-01-10",
                "cost": 5000.00,
                "expected": 2000.00,
                "severity": "high",
            },
            {
                "date": "2023-01-12",
                "cost": 3500.00,
                "expected": 2000.00,
                "severity": "medium",
            },
            {
                "date": "2023-01-14",
                "cost": 2800.00,
                "expected": 2000.00,
                "severity": "low",
            },
        ]

        dto = CostAnalysisResponseDTO(cost_anomalies=anomalies)

        assert len(dto.cost_anomalies) == 3
        assert dto.cost_anomalies[0]["severity"] == "high"
        assert dto.cost_anomalies[1]["severity"] == "medium"
        assert dto.cost_anomalies[2]["severity"] == "low"

    def test_detailed_cost_predictions(self):
        """Test detailed cost predictions."""
        dto = CostAnalysisResponseDTO(
            cost_predictions={
                "next_week": 6000.00,
                "next_month": 25000.00,
                "next_quarter": 75000.00,
                "next_year": 300000.00,
                "confidence_interval": {"lower": 23000.00, "upper": 27000.00},
                "prediction_accuracy": 0.85,
            }
        )

        assert dto.cost_predictions["next_week"] == 6000.00
        assert dto.cost_predictions["next_month"] == 25000.00
        assert dto.cost_predictions["next_quarter"] == 75000.00
        assert dto.cost_predictions["next_year"] == 300000.00
        assert dto.cost_predictions["confidence_interval"]["lower"] == 23000.00
        assert dto.cost_predictions["confidence_interval"]["upper"] == 27000.00
        assert dto.cost_predictions["prediction_accuracy"] == 0.85

    def test_detailed_optimization_potential(self):
        """Test detailed optimization potential."""
        dto = CostAnalysisResponseDTO(
            optimization_potential={
                "monthly_savings": 5000.00,
                "annual_savings": 60000.00,
                "recommendations": 25,
                "high_impact_recommendations": 8,
                "quick_wins": 12,
                "savings_by_type": {
                    "rightsizing": 2000.00,
                    "scheduling": 1500.00,
                    "reserved_instances": 1000.00,
                    "idle_cleanup": 500.00,
                },
                "implementation_effort": "medium",
            }
        )

        assert dto.optimization_potential["monthly_savings"] == 5000.00
        assert dto.optimization_potential["annual_savings"] == 60000.00
        assert dto.optimization_potential["recommendations"] == 25
        assert dto.optimization_potential["high_impact_recommendations"] == 8
        assert dto.optimization_potential["quick_wins"] == 12
        assert dto.optimization_potential["savings_by_type"]["rightsizing"] == 2000.00
        assert dto.optimization_potential["savings_by_type"]["scheduling"] == 1500.00
        assert (
            dto.optimization_potential["savings_by_type"]["reserved_instances"]
            == 1000.00
        )
        assert dto.optimization_potential["savings_by_type"]["idle_cleanup"] == 500.00
        assert dto.optimization_potential["implementation_effort"] == "medium"


class TestOptimizationPlanRequestDTO:
    """Test suite for OptimizationPlanRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization plan request DTO."""
        dto = OptimizationPlanRequestDTO(
            strategy="aggressive",
            tenant_id="tenant-123",
            target_savings_percent=30.0,
            max_risk_level="high",
            max_performance_impact="moderate",
            resource_scope=["compute", "storage", "network"],
            environments=["production", "staging"],
            include_scheduling=True,
            include_rightsizing=True,
            include_instance_optimization=True,
            include_storage_optimization=True,
            include_idle_cleanup=True,
            auto_implement_safe=True,
        )

        assert dto.strategy == "aggressive"
        assert dto.tenant_id == "tenant-123"
        assert dto.target_savings_percent == 30.0
        assert dto.max_risk_level == "high"
        assert dto.max_performance_impact == "moderate"
        assert dto.resource_scope == ["compute", "storage", "network"]
        assert dto.environments == ["production", "staging"]
        assert dto.include_scheduling is True
        assert dto.include_rightsizing is True
        assert dto.include_instance_optimization is True
        assert dto.include_storage_optimization is True
        assert dto.include_idle_cleanup is True
        assert dto.auto_implement_safe is True

    def test_default_values(self):
        """Test default values."""
        dto = OptimizationPlanRequestDTO()

        assert dto.strategy == "balanced"
        assert dto.tenant_id is None
        assert dto.target_savings_percent == 0.2
        assert dto.max_risk_level == "medium"
        assert dto.max_performance_impact == "minimal"
        assert dto.resource_scope == []
        assert dto.environments == []
        assert dto.include_scheduling is True
        assert dto.include_rightsizing is True
        assert dto.include_instance_optimization is True
        assert dto.include_storage_optimization is True
        assert dto.include_idle_cleanup is True
        assert dto.auto_implement_safe is False

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = OptimizationPlanRequestDTO(resource_scope=None, environments=None)

        assert dto.resource_scope == []
        assert dto.environments == []

    def test_different_strategies(self):
        """Test different optimization strategies."""
        strategies = ["conservative", "balanced", "aggressive", "experimental"]

        for strategy in strategies:
            dto = OptimizationPlanRequestDTO(strategy=strategy)
            assert dto.strategy == strategy

    def test_different_target_savings_percents(self):
        """Test different target savings percentages."""
        savings_percents = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        for percent in savings_percents:
            dto = OptimizationPlanRequestDTO(target_savings_percent=percent)
            assert dto.target_savings_percent == percent

    def test_different_risk_levels(self):
        """Test different risk levels."""
        risk_levels = ["low", "medium", "high", "critical"]

        for risk_level in risk_levels:
            dto = OptimizationPlanRequestDTO(max_risk_level=risk_level)
            assert dto.max_risk_level == risk_level

    def test_different_performance_impacts(self):
        """Test different performance impact levels."""
        performance_impacts = ["none", "minimal", "moderate", "significant"]

        for impact in performance_impacts:
            dto = OptimizationPlanRequestDTO(max_performance_impact=impact)
            assert dto.max_performance_impact == impact

    def test_selective_optimization_options(self):
        """Test selective optimization options."""
        dto = OptimizationPlanRequestDTO(
            include_scheduling=False,
            include_rightsizing=True,
            include_instance_optimization=False,
            include_storage_optimization=True,
            include_idle_cleanup=False,
        )

        assert dto.include_scheduling is False
        assert dto.include_rightsizing is True
        assert dto.include_instance_optimization is False
        assert dto.include_storage_optimization is True
        assert dto.include_idle_cleanup is False

    def test_auto_implementation_enabled(self):
        """Test auto implementation enabled."""
        dto = OptimizationPlanRequestDTO(
            strategy="conservative", max_risk_level="low", auto_implement_safe=True
        )

        assert dto.strategy == "conservative"
        assert dto.max_risk_level == "low"
        assert dto.auto_implement_safe is True

    def test_comprehensive_resource_scope(self):
        """Test comprehensive resource scope."""
        dto = OptimizationPlanRequestDTO(
            resource_scope=[
                "compute",
                "storage",
                "network",
                "database",
                "cache",
                "cdn",
                "load_balancer",
                "monitoring",
            ]
        )

        assert len(dto.resource_scope) == 8
        assert "compute" in dto.resource_scope
        assert "storage" in dto.resource_scope
        assert "network" in dto.resource_scope
        assert "database" in dto.resource_scope
        assert "cache" in dto.resource_scope
        assert "cdn" in dto.resource_scope
        assert "load_balancer" in dto.resource_scope
        assert "monitoring" in dto.resource_scope

    def test_multiple_environments(self):
        """Test multiple environments."""
        dto = OptimizationPlanRequestDTO(
            environments=["development", "testing", "staging", "production", "sandbox"]
        )

        assert len(dto.environments) == 5
        assert "development" in dto.environments
        assert "testing" in dto.environments
        assert "staging" in dto.environments
        assert "production" in dto.environments
        assert "sandbox" in dto.environments


class TestRecommendationImplementationDTO:
    """Test suite for RecommendationImplementationDTO."""

    def test_valid_creation(self):
        """Test creating a valid recommendation implementation DTO."""
        scheduled_at = datetime(2023, 1, 20, 2, 0, 0)

        dto = RecommendationImplementationDTO(
            recommendation_id="rec-123",
            implementation_method="automated",
            scheduled_at=scheduled_at,
            dry_run=True,
            confirmation_required=False,
            rollback_plan="Revert instance type to previous configuration",
            notification_contacts=["admin@company.com", "ops@company.com"],
        )

        assert dto.recommendation_id == "rec-123"
        assert dto.implementation_method == "automated"
        assert dto.scheduled_at == scheduled_at
        assert dto.dry_run is True
        assert dto.confirmation_required is False
        assert dto.rollback_plan == "Revert instance type to previous configuration"
        assert dto.notification_contacts == ["admin@company.com", "ops@company.com"]

    def test_default_values(self):
        """Test default values."""
        dto = RecommendationImplementationDTO()

        assert dto.recommendation_id == ""
        assert dto.implementation_method == "manual"
        assert dto.scheduled_at is None
        assert dto.dry_run is False
        assert dto.confirmation_required is True
        assert dto.rollback_plan == ""
        assert dto.notification_contacts == []

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = RecommendationImplementationDTO(notification_contacts=None)

        assert dto.notification_contacts == []

    def test_different_implementation_methods(self):
        """Test different implementation methods."""
        methods = ["manual", "automated", "scheduled"]

        for method in methods:
            dto = RecommendationImplementationDTO(
                recommendation_id="rec-123", implementation_method=method
            )
            assert dto.implementation_method == method

    def test_manual_implementation(self):
        """Test manual implementation configuration."""
        dto = RecommendationImplementationDTO(
            recommendation_id="rec-manual",
            implementation_method="manual",
            confirmation_required=True,
            rollback_plan="Manual rollback instructions provided",
        )

        assert dto.implementation_method == "manual"
        assert dto.confirmation_required is True
        assert dto.rollback_plan == "Manual rollback instructions provided"

    def test_automated_implementation(self):
        """Test automated implementation configuration."""
        dto = RecommendationImplementationDTO(
            recommendation_id="rec-auto",
            implementation_method="automated",
            dry_run=False,
            confirmation_required=False,
            rollback_plan="Automated rollback available",
        )

        assert dto.implementation_method == "automated"
        assert dto.dry_run is False
        assert dto.confirmation_required is False
        assert dto.rollback_plan == "Automated rollback available"

    def test_scheduled_implementation(self):
        """Test scheduled implementation configuration."""
        scheduled_at = datetime(2023, 1, 25, 3, 0, 0)

        dto = RecommendationImplementationDTO(
            recommendation_id="rec-scheduled",
            implementation_method="scheduled",
            scheduled_at=scheduled_at,
            confirmation_required=True,
        )

        assert dto.implementation_method == "scheduled"
        assert dto.scheduled_at == scheduled_at
        assert dto.confirmation_required is True

    def test_dry_run_configuration(self):
        """Test dry run configuration."""
        dto = RecommendationImplementationDTO(
            recommendation_id="rec-dry-run", dry_run=True, confirmation_required=False
        )

        assert dto.dry_run is True
        assert dto.confirmation_required is False

    def test_multiple_notification_contacts(self):
        """Test multiple notification contacts."""
        dto = RecommendationImplementationDTO(
            recommendation_id="rec-notifications",
            notification_contacts=[
                "admin@company.com",
                "ops@company.com",
                "finance@company.com",
                "security@company.com",
            ],
        )

        assert len(dto.notification_contacts) == 4
        assert "admin@company.com" in dto.notification_contacts
        assert "ops@company.com" in dto.notification_contacts
        assert "finance@company.com" in dto.notification_contacts
        assert "security@company.com" in dto.notification_contacts

    def test_complex_rollback_plan(self):
        """Test complex rollback plan."""
        rollback_plan = """
        1. Stop the modified instance
        2. Revert to previous instance type
        3. Restart the instance
        4. Verify application functionality
        5. Update monitoring configurations
        6. Notify stakeholders of rollback completion
        """

        dto = RecommendationImplementationDTO(
            recommendation_id="rec-complex-rollback", rollback_plan=rollback_plan
        )

        assert dto.rollback_plan == rollback_plan


class TestRecommendationImplementationResultDTO:
    """Test suite for RecommendationImplementationResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid recommendation implementation result DTO."""
        started_at = datetime(2023, 1, 20, 2, 0, 0)
        completed_at = datetime(2023, 1, 20, 2, 15, 0)

        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-123",
            recommendation_id="rec-123",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            success=True,
            error_message=None,
            actual_savings=450.00,
            actual_implementation_time="15 minutes",
            performance_impact_observed="none",
            rollback_available=True,
            rollback_instructions="Use rollback script: /scripts/rollback-rec-123.sh",
            validation_results={
                "pre_check": "passed",
                "post_check": "passed",
                "savings_validated": True,
            },
        )

        assert dto.implementation_id == "impl-123"
        assert dto.recommendation_id == "rec-123"
        assert dto.status == "completed"
        assert dto.started_at == started_at
        assert dto.completed_at == completed_at
        assert dto.success is True
        assert dto.error_message is None
        assert dto.actual_savings == 450.00
        assert dto.actual_implementation_time == "15 minutes"
        assert dto.performance_impact_observed == "none"
        assert dto.rollback_available is True
        assert (
            dto.rollback_instructions
            == "Use rollback script: /scripts/rollback-rec-123.sh"
        )
        assert dto.validation_results == {
            "pre_check": "passed",
            "post_check": "passed",
            "savings_validated": True,
        }

    def test_default_values(self):
        """Test default values."""
        dto = RecommendationImplementationResultDTO()

        assert dto.implementation_id == ""
        assert dto.recommendation_id == ""
        assert dto.status == "pending"
        assert dto.started_at is None
        assert dto.completed_at is None
        assert dto.success is False
        assert dto.error_message is None
        assert dto.actual_savings == 0.0
        assert dto.actual_implementation_time is None
        assert dto.performance_impact_observed == "none"
        assert dto.rollback_available is False
        assert dto.rollback_instructions == ""
        assert dto.validation_results == {}

    def test_post_init_dict_initialization(self):
        """Test that __post_init__ properly initializes validation_results."""
        dto = RecommendationImplementationResultDTO(validation_results=None)

        assert dto.validation_results == {}

    def test_different_statuses(self):
        """Test different implementation statuses."""
        statuses = ["pending", "in_progress", "completed", "failed", "rolled_back"]

        for status in statuses:
            dto = RecommendationImplementationResultDTO(
                implementation_id=f"impl-{status}", status=status
            )
            assert dto.status == status

    def test_successful_implementation(self):
        """Test successful implementation."""
        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-success",
            status="completed",
            success=True,
            actual_savings=500.00,
            actual_implementation_time="10 minutes",
            performance_impact_observed="minimal",
        )

        assert dto.status == "completed"
        assert dto.success is True
        assert dto.actual_savings == 500.00
        assert dto.actual_implementation_time == "10 minutes"
        assert dto.performance_impact_observed == "minimal"

    def test_failed_implementation(self):
        """Test failed implementation."""
        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-failed",
            status="failed",
            success=False,
            error_message="Instance failed to resize due to insufficient capacity",
            actual_savings=0.0,
            rollback_available=True,
            rollback_instructions="Rollback to original instance type",
        )

        assert dto.status == "failed"
        assert dto.success is False
        assert (
            dto.error_message
            == "Instance failed to resize due to insufficient capacity"
        )
        assert dto.actual_savings == 0.0
        assert dto.rollback_available is True
        assert dto.rollback_instructions == "Rollback to original instance type"

    def test_in_progress_implementation(self):
        """Test in-progress implementation."""
        started_at = datetime(2023, 1, 20, 2, 0, 0)

        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-in-progress",
            status="in_progress",
            started_at=started_at,
            success=False,
        )

        assert dto.status == "in_progress"
        assert dto.started_at == started_at
        assert dto.completed_at is None
        assert dto.success is False

    def test_rolled_back_implementation(self):
        """Test rolled back implementation."""
        started_at = datetime(2023, 1, 20, 2, 0, 0)
        completed_at = datetime(2023, 1, 20, 2, 45, 0)

        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-rolled-back",
            status="rolled_back",
            started_at=started_at,
            completed_at=completed_at,
            success=False,
            error_message="Performance degradation detected, rolling back",
            actual_savings=0.0,
            performance_impact_observed="significant",
            rollback_available=False,
        )

        assert dto.status == "rolled_back"
        assert dto.started_at == started_at
        assert dto.completed_at == completed_at
        assert dto.success is False
        assert dto.error_message == "Performance degradation detected, rolling back"
        assert dto.actual_savings == 0.0
        assert dto.performance_impact_observed == "significant"
        assert dto.rollback_available is False

    def test_different_performance_impacts(self):
        """Test different performance impact levels."""
        performance_impacts = ["none", "minimal", "moderate", "significant"]

        for impact in performance_impacts:
            dto = RecommendationImplementationResultDTO(
                implementation_id=f"impl-{impact}", performance_impact_observed=impact
            )
            assert dto.performance_impact_observed == impact

    def test_high_savings_implementation(self):
        """Test implementation with high savings."""
        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-high-savings",
            status="completed",
            success=True,
            actual_savings=2500.00,
            actual_implementation_time="30 minutes",
        )

        assert dto.actual_savings == 2500.00
        assert dto.actual_implementation_time == "30 minutes"

    def test_complex_validation_results(self):
        """Test complex validation results."""
        validation_results = {
            "pre_implementation_check": "passed",
            "resource_availability": "confirmed",
            "backup_completed": True,
            "monitoring_updated": True,
            "post_implementation_check": "passed",
            "performance_validation": "passed",
            "cost_savings_validated": True,
            "rollback_tested": True,
            "stakeholder_notification": "sent",
        }

        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-complex-validation",
            validation_results=validation_results,
        )

        assert dto.validation_results == validation_results
        assert dto.validation_results["pre_implementation_check"] == "passed"
        assert dto.validation_results["resource_availability"] == "confirmed"
        assert dto.validation_results["backup_completed"] is True
        assert dto.validation_results["monitoring_updated"] is True
        assert dto.validation_results["post_implementation_check"] == "passed"
        assert dto.validation_results["performance_validation"] == "passed"
        assert dto.validation_results["cost_savings_validated"] is True
        assert dto.validation_results["rollback_tested"] is True
        assert dto.validation_results["stakeholder_notification"] == "sent"

    def test_long_implementation_time(self):
        """Test long implementation time."""
        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-long-time",
            actual_implementation_time="2 hours 30 minutes",
        )

        assert dto.actual_implementation_time == "2 hours 30 minutes"

    def test_detailed_error_message(self):
        """Test detailed error message."""
        error_message = """
        Implementation failed during step 3 of 5:
        - Step 1: Instance stopped successfully
        - Step 2: Backup completed successfully
        - Step 3: Instance resize failed - insufficient capacity in availability zone
        - Error Code: InsufficientInstanceCapacity
        - Recommendation: Try again during off-peak hours or select different AZ
        """

        dto = RecommendationImplementationResultDTO(
            implementation_id="impl-detailed-error",
            status="failed",
            success=False,
            error_message=error_message,
        )

        assert dto.error_message == error_message


class TestBudgetAlertDTO:
    """Test suite for BudgetAlertDTO."""

    def test_valid_creation(self):
        """Test creating a valid budget alert DTO."""
        triggered_at = datetime(2023, 1, 15, 14, 30, 0)

        dto = BudgetAlertDTO(
            alert_id="alert-123",
            budget_id="budget-456",
            budget_name="Engineering Q1 Budget",
            alert_type="budget_threshold",
            threshold=0.8,
            current_utilization=0.85,
            current_spend=8500.00,
            budget_limit=10000.00,
            severity="high",
            triggered_at=triggered_at,
            days_until_exhausted=5,
            recommended_actions=[
                "Review high-cost resources",
                "Pause non-critical workloads",
            ],
        )

        assert dto.alert_id == "alert-123"
        assert dto.budget_id == "budget-456"
        assert dto.budget_name == "Engineering Q1 Budget"
        assert dto.alert_type == "budget_threshold"
        assert dto.threshold == 0.8
        assert dto.current_utilization == 0.85
        assert dto.current_spend == 8500.00
        assert dto.budget_limit == 10000.00
        assert dto.severity == "high"
        assert dto.triggered_at == triggered_at
        assert dto.days_until_exhausted == 5
        assert dto.recommended_actions == [
            "Review high-cost resources",
            "Pause non-critical workloads",
        ]

    def test_default_values(self):
        """Test default values."""
        dto = BudgetAlertDTO()

        assert dto.alert_id == ""
        assert dto.budget_id == ""
        assert dto.budget_name == ""
        assert dto.alert_type == "budget_threshold"
        assert dto.threshold == 0.0
        assert dto.current_utilization == 0.0
        assert dto.current_spend == 0.0
        assert dto.budget_limit == 0.0
        assert dto.severity == "medium"
        assert dto.triggered_at is None
        assert dto.days_until_exhausted is None
        assert dto.recommended_actions == []

    def test_post_init_list_initialization(self):
        """Test that __post_init__ properly initializes lists."""
        dto = BudgetAlertDTO(recommended_actions=None)

        assert dto.recommended_actions == []

    def test_different_alert_types(self):
        """Test different alert types."""
        alert_types = [
            "budget_threshold",
            "spending_anomaly",
            "forecast_exceeded",
            "variance_detected",
        ]

        for alert_type in alert_types:
            dto = BudgetAlertDTO(alert_id=f"alert-{alert_type}", alert_type=alert_type)
            assert dto.alert_type == alert_type

    def test_different_severities(self):
        """Test different severity levels."""
        severities = ["low", "medium", "high", "critical"]

        for severity in severities:
            dto = BudgetAlertDTO(alert_id=f"alert-{severity}", severity=severity)
            assert dto.severity == severity

    def test_different_thresholds(self):
        """Test different threshold values."""
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for threshold in thresholds:
            dto = BudgetAlertDTO(alert_id=f"alert-{threshold}", threshold=threshold)
            assert dto.threshold == threshold

    def test_high_utilization_alert(self):
        """Test high utilization alert."""
        dto = BudgetAlertDTO(
            alert_id="alert-high-util",
            threshold=0.9,
            current_utilization=0.95,
            current_spend=95000.00,
            budget_limit=100000.00,
            severity="critical",
            days_until_exhausted=2,
        )

        assert dto.threshold == 0.9
        assert dto.current_utilization == 0.95
        assert dto.current_spend == 95000.00
        assert dto.budget_limit == 100000.00
        assert dto.severity == "critical"
        assert dto.days_until_exhausted == 2

    def test_budget_exceeded_alert(self):
        """Test budget exceeded alert."""
        dto = BudgetAlertDTO(
            alert_id="alert-exceeded",
            threshold=1.0,
            current_utilization=1.15,
            current_spend=115000.00,
            budget_limit=100000.00,
            severity="critical",
            days_until_exhausted=0,
        )

        assert dto.threshold == 1.0
        assert dto.current_utilization == 1.15
        assert dto.current_spend == 115000.00
        assert dto.budget_limit == 100000.00
        assert dto.severity == "critical"
        assert dto.days_until_exhausted == 0

    def test_early_warning_alert(self):
        """Test early warning alert."""
        dto = BudgetAlertDTO(
            alert_id="alert-early-warning",
            threshold=0.5,
            current_utilization=0.52,
            current_spend=5200.00,
            budget_limit=10000.00,
            severity="low",
            days_until_exhausted=15,
        )

        assert dto.threshold == 0.5
        assert dto.current_utilization == 0.52
        assert dto.current_spend == 5200.00
        assert dto.budget_limit == 10000.00
        assert dto.severity == "low"
        assert dto.days_until_exhausted == 15

    def test_many_recommended_actions(self):
        """Test many recommended actions."""
        actions = [
            "Review and analyze top spending resources",
            "Implement cost optimization recommendations",
            "Pause non-critical development workloads",
            "Scale down staging environments",
            "Review and optimize storage usage",
            "Implement resource scheduling",
            "Consider reserved instance purchases",
            "Review and remove unused resources",
            "Implement auto-scaling policies",
            "Contact finance team for budget review",
        ]

        dto = BudgetAlertDTO(alert_id="alert-many-actions", recommended_actions=actions)

        assert len(dto.recommended_actions) == 10
        assert "Review and analyze top spending resources" in dto.recommended_actions
        assert "Implement cost optimization recommendations" in dto.recommended_actions
        assert "Contact finance team for budget review" in dto.recommended_actions

    def test_long_budget_name(self):
        """Test long budget name."""
        long_name = "Engineering Team Q1 2023 Multi-Cloud Infrastructure Budget Including Development, Staging, and Production Environments"

        dto = BudgetAlertDTO(alert_id="alert-long-name", budget_name=long_name)

        assert dto.budget_name == long_name

    def test_large_budget_values(self):
        """Test large budget values."""
        dto = BudgetAlertDTO(
            alert_id="alert-large-budget",
            current_spend=850000.00,
            budget_limit=1000000.00,
            current_utilization=0.85,
        )

        assert dto.current_spend == 850000.00
        assert dto.budget_limit == 1000000.00
        assert dto.current_utilization == 0.85


class TestResourceSummaryDTO:
    """Test suite for ResourceSummaryDTO."""

    def test_valid_creation(self):
        """Test creating a valid resource summary DTO."""
        dto = ResourceSummaryDTO(
            total_resources=250,
            total_monthly_cost=45000.00,
            resource_breakdown={
                "compute": {"count": 150, "cost": 30000.00},
                "storage": {"count": 80, "cost": 10000.00},
                "network": {"count": 20, "cost": 5000.00},
            },
            optimization_summary={
                "idle_resources": 25,
                "underutilized_resources": 40,
                "optimization_opportunities": 15,
            },
            cost_efficiency_score=0.75,
            recommendations_available=18,
            potential_monthly_savings=8000.00,
        )

        assert dto.total_resources == 250
        assert dto.total_monthly_cost == 45000.00
        assert dto.resource_breakdown["compute"]["count"] == 150
        assert dto.resource_breakdown["compute"]["cost"] == 30000.00
        assert dto.resource_breakdown["storage"]["count"] == 80
        assert dto.resource_breakdown["storage"]["cost"] == 10000.00
        assert dto.resource_breakdown["network"]["count"] == 20
        assert dto.resource_breakdown["network"]["cost"] == 5000.00
        assert dto.optimization_summary["idle_resources"] == 25
        assert dto.optimization_summary["underutilized_resources"] == 40
        assert dto.optimization_summary["optimization_opportunities"] == 15
        assert dto.cost_efficiency_score == 0.75
        assert dto.recommendations_available == 18
        assert dto.potential_monthly_savings == 8000.00

    def test_default_values(self):
        """Test default values."""
        dto = ResourceSummaryDTO()

        assert dto.total_resources == 0
        assert dto.total_monthly_cost == 0.0
        assert dto.resource_breakdown == {}
        assert dto.optimization_summary == {}
        assert dto.cost_efficiency_score == 0.0
        assert dto.recommendations_available == 0
        assert dto.potential_monthly_savings == 0.0

    def test_post_init_dict_initialization(self):
        """Test that __post_init__ properly initializes dicts."""
        dto = ResourceSummaryDTO(resource_breakdown=None, optimization_summary=None)

        assert dto.resource_breakdown == {}
        assert dto.optimization_summary == {}

    def test_large_scale_summary(self):
        """Test large scale resource summary."""
        dto = ResourceSummaryDTO(
            total_resources=5000,
            total_monthly_cost=1500000.00,
            cost_efficiency_score=0.85,
            recommendations_available=200,
            potential_monthly_savings=250000.00,
        )

        assert dto.total_resources == 5000
        assert dto.total_monthly_cost == 1500000.00
        assert dto.cost_efficiency_score == 0.85
        assert dto.recommendations_available == 200
        assert dto.potential_monthly_savings == 250000.00

    def test_detailed_resource_breakdown(self):
        """Test detailed resource breakdown."""
        breakdown = {
            "compute": {"count": 200, "cost": 50000.00},
            "storage": {"count": 150, "cost": 25000.00},
            "network": {"count": 50, "cost": 10000.00},
            "database": {"count": 30, "cost": 15000.00},
            "cache": {"count": 20, "cost": 5000.00},
            "cdn": {"count": 10, "cost": 3000.00},
            "load_balancer": {"count": 15, "cost": 2000.00},
            "monitoring": {"count": 25, "cost": 1500.00},
        }

        dto = ResourceSummaryDTO(resource_breakdown=breakdown)

        assert len(dto.resource_breakdown) == 8
        assert dto.resource_breakdown["compute"]["count"] == 200
        assert dto.resource_breakdown["compute"]["cost"] == 50000.00
        assert dto.resource_breakdown["storage"]["count"] == 150
        assert dto.resource_breakdown["storage"]["cost"] == 25000.00
        assert dto.resource_breakdown["database"]["count"] == 30
        assert dto.resource_breakdown["database"]["cost"] == 15000.00
        assert dto.resource_breakdown["monitoring"]["count"] == 25
        assert dto.resource_breakdown["monitoring"]["cost"] == 1500.00

    def test_detailed_optimization_summary(self):
        """Test detailed optimization summary."""
        optimization = {
            "idle_resources": 50,
            "underutilized_resources": 80,
            "oversized_resources": 30,
            "untagged_resources": 25,
            "duplicate_resources": 10,
            "optimization_opportunities": 35,
            "quick_wins": 20,
            "high_impact_optimizations": 15,
        }

        dto = ResourceSummaryDTO(optimization_summary=optimization)

        assert len(dto.optimization_summary) == 8
        assert dto.optimization_summary["idle_resources"] == 50
        assert dto.optimization_summary["underutilized_resources"] == 80
        assert dto.optimization_summary["oversized_resources"] == 30
        assert dto.optimization_summary["untagged_resources"] == 25
        assert dto.optimization_summary["duplicate_resources"] == 10
        assert dto.optimization_summary["optimization_opportunities"] == 35
        assert dto.optimization_summary["quick_wins"] == 20
        assert dto.optimization_summary["high_impact_optimizations"] == 15

    def test_high_efficiency_score(self):
        """Test high efficiency score."""
        dto = ResourceSummaryDTO(
            cost_efficiency_score=0.95,
            recommendations_available=5,
            potential_monthly_savings=1000.00,
        )

        assert dto.cost_efficiency_score == 0.95
        assert dto.recommendations_available == 5
        assert dto.potential_monthly_savings == 1000.00

    def test_low_efficiency_score(self):
        """Test low efficiency score."""
        dto = ResourceSummaryDTO(
            cost_efficiency_score=0.35,
            recommendations_available=50,
            potential_monthly_savings=25000.00,
        )

        assert dto.cost_efficiency_score == 0.35
        assert dto.recommendations_available == 50
        assert dto.potential_monthly_savings == 25000.00

    def test_zero_resources(self):
        """Test zero resources case."""
        dto = ResourceSummaryDTO(
            total_resources=0,
            total_monthly_cost=0.0,
            cost_efficiency_score=0.0,
            recommendations_available=0,
            potential_monthly_savings=0.0,
        )

        assert dto.total_resources == 0
        assert dto.total_monthly_cost == 0.0
        assert dto.cost_efficiency_score == 0.0
        assert dto.recommendations_available == 0
        assert dto.potential_monthly_savings == 0.0


class TestServiceMetricsDTO:
    """Test suite for ServiceMetricsDTO."""

    def test_valid_creation(self):
        """Test creating a valid service metrics DTO."""
        last_updated = datetime(2023, 1, 15, 10, 0, 0)

        dto = ServiceMetricsDTO(
            total_resources=500,
            total_monthly_cost=125000.00,
            total_savings_identified=25000.00,
            recommendations_generated=150,
            recommendations_implemented=120,
            optimization_plans=15,
            budgets=8,
            avg_cost_per_resource=250.00,
            savings_rate=0.20,
            last_updated=last_updated,
        )

        assert dto.total_resources == 500
        assert dto.total_monthly_cost == 125000.00
        assert dto.total_savings_identified == 25000.00
        assert dto.recommendations_generated == 150
        assert dto.recommendations_implemented == 120
        assert dto.optimization_plans == 15
        assert dto.budgets == 8
        assert dto.avg_cost_per_resource == 250.00
        assert dto.savings_rate == 0.20
        assert dto.last_updated == last_updated

    def test_default_values(self):
        """Test default values."""
        dto = ServiceMetricsDTO()

        assert dto.total_resources == 0
        assert dto.total_monthly_cost == 0.0
        assert dto.total_savings_identified == 0.0
        assert dto.recommendations_generated == 0
        assert dto.recommendations_implemented == 0
        assert dto.optimization_plans == 0
        assert dto.budgets == 0
        assert dto.avg_cost_per_resource == 0.0
        assert dto.savings_rate == 0.0
        assert dto.last_updated is None

    def test_large_scale_metrics(self):
        """Test large scale service metrics."""
        dto = ServiceMetricsDTO(
            total_resources=10000,
            total_monthly_cost=2500000.00,
            total_savings_identified=500000.00,
            recommendations_generated=1500,
            recommendations_implemented=1200,
            optimization_plans=50,
            budgets=25,
            avg_cost_per_resource=250.00,
            savings_rate=0.20,
        )

        assert dto.total_resources == 10000
        assert dto.total_monthly_cost == 2500000.00
        assert dto.total_savings_identified == 500000.00
        assert dto.recommendations_generated == 1500
        assert dto.recommendations_implemented == 1200
        assert dto.optimization_plans == 50
        assert dto.budgets == 25
        assert dto.avg_cost_per_resource == 250.00
        assert dto.savings_rate == 0.20

    def test_high_savings_rate(self):
        """Test high savings rate."""
        dto = ServiceMetricsDTO(
            total_monthly_cost=100000.00,
            total_savings_identified=40000.00,
            savings_rate=0.40,
        )

        assert dto.total_monthly_cost == 100000.00
        assert dto.total_savings_identified == 40000.00
        assert dto.savings_rate == 0.40

    def test_high_implementation_rate(self):
        """Test high implementation rate."""
        dto = ServiceMetricsDTO(
            recommendations_generated=200, recommendations_implemented=190
        )

        assert dto.recommendations_generated == 200
        assert dto.recommendations_implemented == 190
        # Implementation rate would be 95%

    def test_low_cost_per_resource(self):
        """Test low cost per resource."""
        dto = ServiceMetricsDTO(
            total_resources=1000,
            total_monthly_cost=50000.00,
            avg_cost_per_resource=50.00,
        )

        assert dto.total_resources == 1000
        assert dto.total_monthly_cost == 50000.00
        assert dto.avg_cost_per_resource == 50.00

    def test_high_cost_per_resource(self):
        """Test high cost per resource."""
        dto = ServiceMetricsDTO(
            total_resources=100,
            total_monthly_cost=100000.00,
            avg_cost_per_resource=1000.00,
        )

        assert dto.total_resources == 100
        assert dto.total_monthly_cost == 100000.00
        assert dto.avg_cost_per_resource == 1000.00

    def test_many_optimization_plans(self):
        """Test many optimization plans."""
        dto = ServiceMetricsDTO(optimization_plans=100, budgets=50)

        assert dto.optimization_plans == 100
        assert dto.budgets == 50

    def test_zero_savings_rate(self):
        """Test zero savings rate."""
        dto = ServiceMetricsDTO(
            total_monthly_cost=50000.00, total_savings_identified=0.0, savings_rate=0.0
        )

        assert dto.total_monthly_cost == 50000.00
        assert dto.total_savings_identified == 0.0
        assert dto.savings_rate == 0.0

    def test_perfect_implementation_rate(self):
        """Test perfect implementation rate."""
        dto = ServiceMetricsDTO(
            recommendations_generated=100, recommendations_implemented=100
        )

        assert dto.recommendations_generated == 100
        assert dto.recommendations_implemented == 100

    def test_recent_update(self):
        """Test recent update timestamp."""
        recent_time = datetime(2023, 1, 20, 15, 30, 45)

        dto = ServiceMetricsDTO(last_updated=recent_time)

        assert dto.last_updated == recent_time
