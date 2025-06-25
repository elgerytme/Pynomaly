"""Domain entities for multi-tenant architecture."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class TenantStatus(Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    PENDING_ACTIVATION = "pending_activation"


class SubscriptionTier(Enum):
    """Subscription tier enumeration."""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class ResourceQuotaType(Enum):
    """Resource quota types."""

    CPU_HOURS = "cpu_hours"
    MEMORY_GB = "memory_gb"
    STORAGE_GB = "storage_gb"
    API_REQUESTS = "api_requests"
    CONCURRENT_JOBS = "concurrent_jobs"
    MODELS = "models"
    DATASETS = "datasets"
    USERS = "users"


@dataclass
class ResourceQuota:
    """Resource quota for a tenant."""

    quota_type: ResourceQuotaType
    limit: int
    used: int = 0
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime | None = None
    is_unlimited: bool = False

    @property
    def remaining(self) -> int:
        """Get remaining quota."""
        if self.is_unlimited:
            return float("inf")
        return max(0, self.limit - self.used)

    @property
    def usage_percentage(self) -> float:
        """Get usage percentage."""
        if self.is_unlimited or self.limit == 0:
            return 0.0
        return (self.used / self.limit) * 100

    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        if self.is_unlimited:
            return False
        return self.used >= self.limit

    def can_consume(self, amount: int) -> bool:
        """Check if can consume additional resources."""
        if self.is_unlimited:
            return True
        return (self.used + amount) <= self.limit


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration."""

    max_concurrent_jobs: int = 5
    max_model_size_mb: int = 1000
    allowed_algorithms: set[str] = field(default_factory=set)
    allowed_data_formats: set[str] = field(
        default_factory=lambda: {"csv", "json", "parquet"}
    )
    enable_auto_scaling: bool = True
    enable_gpu_access: bool = False
    enable_advanced_analytics: bool = True
    data_retention_days: int = 365
    backup_enabled: bool = True
    monitoring_level: str = "standard"  # basic, standard, advanced
    custom_settings: dict[str, any] = field(default_factory=dict)


@dataclass
class Tenant:
    """Multi-tenant architecture tenant entity."""

    tenant_id: UUID = field(default_factory=uuid4)
    name: str = ""
    display_name: str = ""
    description: str = ""
    status: TenantStatus = TenantStatus.PENDING_ACTIVATION
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE

    # Contact and billing information
    contact_email: str = ""
    admin_user_id: UUID | None = None
    billing_contact: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: datetime | None = None
    last_activity: datetime | None = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Resource management
    resource_quotas: dict[ResourceQuotaType, ResourceQuota] = field(
        default_factory=dict
    )
    configuration: TenantConfiguration = field(default_factory=TenantConfiguration)

    # Isolation and security
    database_schema: str = ""
    encryption_key_id: str = ""
    network_isolation_config: dict[str, any] = field(default_factory=dict)

    # Usage tracking
    total_api_requests: int = 0
    total_cpu_hours: float = 0.0
    total_storage_gb: float = 0.0
    last_billing_date: datetime | None = None

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.database_schema:
            self.database_schema = f"tenant_{str(self.tenant_id).replace('-', '_')}"

        # Initialize default quotas based on subscription tier
        if not self.resource_quotas:
            self._initialize_default_quotas()

    def _initialize_default_quotas(self):
        """Initialize default resource quotas based on subscription tier."""
        quota_limits = {
            SubscriptionTier.FREE: {
                ResourceQuotaType.CPU_HOURS: 10,
                ResourceQuotaType.MEMORY_GB: 2,
                ResourceQuotaType.STORAGE_GB: 5,
                ResourceQuotaType.API_REQUESTS: 1000,
                ResourceQuotaType.CONCURRENT_JOBS: 1,
                ResourceQuotaType.MODELS: 5,
                ResourceQuotaType.DATASETS: 10,
                ResourceQuotaType.USERS: 1,
            },
            SubscriptionTier.BASIC: {
                ResourceQuotaType.CPU_HOURS: 100,
                ResourceQuotaType.MEMORY_GB: 8,
                ResourceQuotaType.STORAGE_GB: 50,
                ResourceQuotaType.API_REQUESTS: 10000,
                ResourceQuotaType.CONCURRENT_JOBS: 3,
                ResourceQuotaType.MODELS: 25,
                ResourceQuotaType.DATASETS: 50,
                ResourceQuotaType.USERS: 5,
            },
            SubscriptionTier.PROFESSIONAL: {
                ResourceQuotaType.CPU_HOURS: 500,
                ResourceQuotaType.MEMORY_GB: 32,
                ResourceQuotaType.STORAGE_GB: 200,
                ResourceQuotaType.API_REQUESTS: 100000,
                ResourceQuotaType.CONCURRENT_JOBS: 10,
                ResourceQuotaType.MODELS: 100,
                ResourceQuotaType.DATASETS: 200,
                ResourceQuotaType.USERS: 25,
            },
            SubscriptionTier.ENTERPRISE: {
                ResourceQuotaType.CPU_HOURS: -1,  # Unlimited
                ResourceQuotaType.MEMORY_GB: -1,
                ResourceQuotaType.STORAGE_GB: -1,
                ResourceQuotaType.API_REQUESTS: -1,
                ResourceQuotaType.CONCURRENT_JOBS: 50,
                ResourceQuotaType.MODELS: -1,
                ResourceQuotaType.DATASETS: -1,
                ResourceQuotaType.USERS: 100,
            },
        }

        tier_limits = quota_limits.get(
            self.subscription_tier, quota_limits[SubscriptionTier.FREE]
        )

        for quota_type, limit in tier_limits.items():
            is_unlimited = limit == -1
            actual_limit = 0 if is_unlimited else limit

            self.resource_quotas[quota_type] = ResourceQuota(
                quota_type=quota_type, limit=actual_limit, is_unlimited=is_unlimited
            )

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def can_access_feature(self, feature_name: str) -> bool:
        """Check if tenant can access a specific feature based on subscription tier."""
        feature_access = {
            SubscriptionTier.FREE: {"basic_detection", "csv_import", "json_export"},
            SubscriptionTier.BASIC: {
                "basic_detection",
                "csv_import",
                "json_export",
                "advanced_algorithms",
                "batch_processing",
                "api_access",
            },
            SubscriptionTier.PROFESSIONAL: {
                "basic_detection",
                "csv_import",
                "json_export",
                "advanced_algorithms",
                "batch_processing",
                "api_access",
                "real_time_detection",
                "custom_models",
                "advanced_analytics",
                "webhooks",
            },
            SubscriptionTier.ENTERPRISE: {
                "basic_detection",
                "csv_import",
                "json_export",
                "advanced_algorithms",
                "batch_processing",
                "api_access",
                "real_time_detection",
                "custom_models",
                "advanced_analytics",
                "webhooks",
                "sso",
                "advanced_security",
                "dedicated_resources",
                "priority_support",
                "custom_integrations",
            },
        }

        allowed_features = feature_access.get(self.subscription_tier, set())
        return feature_name in allowed_features

    def get_quota(self, quota_type: ResourceQuotaType) -> ResourceQuota | None:
        """Get resource quota by type."""
        return self.resource_quotas.get(quota_type)

    def consume_quota(self, quota_type: ResourceQuotaType, amount: int) -> bool:
        """Consume resource quota."""
        quota = self.get_quota(quota_type)
        if not quota:
            return False

        if not quota.can_consume(amount):
            return False

        quota.used += amount
        self.updated_at = datetime.utcnow()
        return True

    def reset_quota(self, quota_type: ResourceQuotaType):
        """Reset quota usage for a new billing period."""
        quota = self.get_quota(quota_type)
        if quota:
            quota.used = 0
            quota.period_start = datetime.utcnow()
            self.updated_at = datetime.utcnow()

    def upgrade_subscription(self, new_tier: SubscriptionTier):
        """Upgrade subscription tier and update quotas."""
        self.subscription_tier = new_tier
        self.updated_at = datetime.utcnow()

        # Reinitialize quotas for new tier
        self._initialize_default_quotas()

        # Update configuration based on new tier
        if new_tier == SubscriptionTier.ENTERPRISE:
            self.configuration.enable_gpu_access = True
            self.configuration.enable_auto_scaling = True
            self.configuration.monitoring_level = "advanced"

    def get_usage_summary(self) -> dict[str, any]:
        """Get usage summary for the tenant."""
        summary = {
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "subscription_tier": self.subscription_tier.value,
            "status": self.status.value,
            "quotas": {},
            "overall_usage_percentage": 0.0,
        }

        total_usage = 0.0
        quota_count = 0

        for quota_type, quota in self.resource_quotas.items():
            quota_info = {
                "limit": quota.limit if not quota.is_unlimited else "unlimited",
                "used": quota.used,
                "remaining": quota.remaining if not quota.is_unlimited else "unlimited",
                "usage_percentage": quota.usage_percentage,
                "is_exceeded": quota.is_exceeded(),
            }
            summary["quotas"][quota_type.value] = quota_info

            if not quota.is_unlimited:
                total_usage += quota.usage_percentage
                quota_count += 1

        if quota_count > 0:
            summary["overall_usage_percentage"] = total_usage / quota_count

        return summary

    def to_dict(self) -> dict[str, any]:
        """Convert tenant to dictionary representation."""
        return {
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "status": self.status.value,
            "subscription_tier": self.subscription_tier.value,
            "contact_email": self.contact_email,
            "admin_user_id": str(self.admin_user_id) if self.admin_user_id else None,
            "billing_contact": self.billing_contact,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat()
            if self.activated_at
            else None,
            "last_activity": self.last_activity.isoformat()
            if self.last_activity
            else None,
            "updated_at": self.updated_at.isoformat(),
            "database_schema": self.database_schema,
            "encryption_key_id": self.encryption_key_id,
            "network_isolation_config": self.network_isolation_config,
            "total_api_requests": self.total_api_requests,
            "total_cpu_hours": self.total_cpu_hours,
            "total_storage_gb": self.total_storage_gb,
            "last_billing_date": self.last_billing_date.isoformat()
            if self.last_billing_date
            else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "configuration": {
                "max_concurrent_jobs": self.configuration.max_concurrent_jobs,
                "max_model_size_mb": self.configuration.max_model_size_mb,
                "allowed_algorithms": list(self.configuration.allowed_algorithms),
                "allowed_data_formats": list(self.configuration.allowed_data_formats),
                "enable_auto_scaling": self.configuration.enable_auto_scaling,
                "enable_gpu_access": self.configuration.enable_gpu_access,
                "enable_advanced_analytics": self.configuration.enable_advanced_analytics,
                "data_retention_days": self.configuration.data_retention_days,
                "backup_enabled": self.configuration.backup_enabled,
                "monitoring_level": self.configuration.monitoring_level,
                "custom_settings": self.configuration.custom_settings,
            },
            "resource_quotas": {
                quota_type.value: {
                    "limit": quota.limit,
                    "used": quota.used,
                    "period_start": quota.period_start.isoformat(),
                    "period_end": quota.period_end.isoformat()
                    if quota.period_end
                    else None,
                    "is_unlimited": quota.is_unlimited,
                }
                for quota_type, quota in self.resource_quotas.items()
            },
        }
