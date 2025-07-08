"""Enterprise multi-tenant management system."""

import asyncio
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pynomaly.shared.config import Config

# from ..monitoring.opentelemetry_service import get_telemetry_service


# Simple stub for monitoring
def get_telemetry_service():
    """Simple stub for monitoring."""
    return None


from ..compliance.audit_system import EventType, Severity, get_audit_system

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DEACTIVATED = "deactivated"
    MIGRATING = "migrating"


class TenantTier(Enum):
    """Tenant service tiers."""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantConfiguration:
    """Tenant configuration settings."""

    # Resource limits
    max_cpu_cores: int = 2
    max_memory_gb: int = 4
    max_storage_gb: int = 10
    max_bandwidth_mbps: int = 100
    max_requests_per_minute: int = 1000
    max_users: int = 10
    max_models: int = 5
    max_datasets: int = 20

    # Feature flags
    features_enabled: set[str] = field(default_factory=set)
    api_access_enabled: bool = True
    ui_access_enabled: bool = True
    export_enabled: bool = True
    advanced_analytics: bool = False
    custom_algorithms: bool = False
    priority_support: bool = False

    # Data retention and compliance
    data_retention_days: int = 90
    compliance_frameworks: set[str] = field(default_factory=set)
    encryption_required: bool = True
    audit_logging_enabled: bool = True

    # Geographic and regulatory
    allowed_regions: set[str] = field(default_factory=lambda: {"us-east-1"})
    data_residency_requirements: set[str] = field(default_factory=set)
    regulatory_compliance: set[str] = field(default_factory=set)


@dataclass
class Tenant:
    """Tenant entity."""

    tenant_id: str
    name: str
    display_name: str
    description: str
    status: TenantStatus
    tier: TenantTier
    configuration: TenantConfiguration
    created_at: datetime
    updated_at: datetime
    owner_user_id: str
    contact_email: str

    # Billing and subscription
    subscription_id: str | None = None
    billing_address: dict[str, str] | None = None
    payment_method_id: str | None = None

    # Security
    api_key: str | None = None
    webhook_secret: str | None = None
    allowed_ip_ranges: set[str] = field(default_factory=set)

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    custom_attributes: dict[str, Any] = field(default_factory=dict)

    # Usage metrics
    last_activity: datetime | None = None
    total_requests: int = 0
    total_data_processed_gb: float = 0.0

    def __post_init__(self):
        """Generate API key if not provided."""
        if not self.api_key:
            self.api_key = self._generate_api_key()
        if not self.webhook_secret:
            self.webhook_secret = secrets.token_urlsafe(32)

    def _generate_api_key(self) -> str:
        """Generate secure API key for tenant."""
        prefix = f"pyn_{self.tenant_id[:8]}"
        suffix = secrets.token_urlsafe(32)
        return f"{prefix}_{suffix}"

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has specific feature enabled."""
        return feature in self.configuration.features_enabled

    def update_last_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
        self.updated_at = datetime.now()


class TenantManager:
    """Enterprise multi-tenant management system."""

    def __init__(self, config: Config | None = None):
        """Initialize tenant manager."""
        self.config = config or Config()
        self.telemetry = get_telemetry_service()
        self.audit_system = get_audit_system()

        # Storage
        self.tenants: dict[str, Tenant] = {}
        self.tenant_by_api_key: dict[str, str] = {}  # api_key -> tenant_id

        # Default configurations by tier
        self.tier_configurations = self._initialize_tier_configurations()

        # Tenant isolation settings
        self.enable_strict_isolation = self.config.get(
            "multitenancy.strict_isolation", True
        )
        self.enable_resource_monitoring = self.config.get(
            "multitenancy.resource_monitoring", True
        )

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_active = False

        logger.info("Tenant manager initialized")

    def _initialize_tier_configurations(self) -> dict[TenantTier, TenantConfiguration]:
        """Initialize default configurations for each tier."""
        return {
            TenantTier.FREE: TenantConfiguration(
                max_cpu_cores=1,
                max_memory_gb=1,
                max_storage_gb=1,
                max_bandwidth_mbps=10,
                max_requests_per_minute=100,
                max_users=1,
                max_models=1,
                max_datasets=5,
                features_enabled={"basic_detection"},
                advanced_analytics=False,
                custom_algorithms=False,
                priority_support=False,
                data_retention_days=30,
            ),
            TenantTier.BASIC: TenantConfiguration(
                max_cpu_cores=2,
                max_memory_gb=4,
                max_storage_gb=10,
                max_bandwidth_mbps=50,
                max_requests_per_minute=500,
                max_users=5,
                max_models=3,
                max_datasets=10,
                features_enabled={"basic_detection", "data_export", "basic_analytics"},
                advanced_analytics=False,
                custom_algorithms=False,
                priority_support=False,
                data_retention_days=90,
            ),
            TenantTier.PROFESSIONAL: TenantConfiguration(
                max_cpu_cores=4,
                max_memory_gb=8,
                max_storage_gb=50,
                max_bandwidth_mbps=100,
                max_requests_per_minute=2000,
                max_users=20,
                max_models=10,
                max_datasets=50,
                features_enabled={
                    "basic_detection",
                    "advanced_detection",
                    "data_export",
                    "basic_analytics",
                    "advanced_analytics",
                    "model_training",
                },
                advanced_analytics=True,
                custom_algorithms=False,
                priority_support=True,
                data_retention_days=365,
            ),
            TenantTier.ENTERPRISE: TenantConfiguration(
                max_cpu_cores=16,
                max_memory_gb=32,
                max_storage_gb=500,
                max_bandwidth_mbps=1000,
                max_requests_per_minute=10000,
                max_users=100,
                max_models=50,
                max_datasets=200,
                features_enabled={
                    "basic_detection",
                    "advanced_detection",
                    "ensemble_methods",
                    "data_export",
                    "basic_analytics",
                    "advanced_analytics",
                    "model_training",
                    "automl",
                    "explainable_ai",
                    "drift_detection",
                    "real_time_processing",
                    "custom_integrations",
                },
                advanced_analytics=True,
                custom_algorithms=True,
                priority_support=True,
                data_retention_days=2555,  # 7 years
                compliance_frameworks={"gdpr", "hipaa", "sox"},
                allowed_regions={"us-east-1", "us-west-2", "eu-west-1"},
            ),
        }

    async def create_tenant(
        self,
        name: str,
        display_name: str,
        description: str,
        tier: TenantTier,
        owner_user_id: str,
        contact_email: str,
        custom_config: TenantConfiguration | None = None,
    ) -> Tenant:
        """Create a new tenant."""
        try:
            tenant_id = str(uuid.uuid4())

            # Get configuration for tier
            if custom_config:
                configuration = custom_config
            else:
                configuration = self.tier_configurations[tier]

            # Create tenant
            tenant = Tenant(
                tenant_id=tenant_id,
                name=name,
                display_name=display_name,
                description=description,
                status=TenantStatus.PENDING,
                tier=tier,
                configuration=configuration,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                owner_user_id=owner_user_id,
                contact_email=contact_email,
            )

            # Store tenant
            self.tenants[tenant_id] = tenant
            self.tenant_by_api_key[tenant.api_key] = tenant_id

            # Audit log
            await self.audit_system.log_event(
                event_type=EventType.SYSTEM_CONFIG_CHANGE,
                user_id=owner_user_id,
                action="create_tenant",
                outcome="success",
                severity=Severity.MEDIUM,
                details={
                    "tenant_id": tenant_id,
                    "tenant_name": name,
                    "tier": tier.value,
                },
                tenant_id=tenant_id,
            )

            # Record metrics
            self.telemetry.record_detection_metrics(
                duration=0,
                anomaly_count=1,
                algorithm="tenant_management",
                tenant_id=tenant_id,
            )

            logger.info(f"Created tenant {tenant_id} ({name}) with tier {tier.value}")
            return tenant

        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    async def get_tenant_by_api_key(self, api_key: str) -> Tenant | None:
        """Get tenant by API key."""
        tenant_id = self.tenant_by_api_key.get(api_key)
        if tenant_id:
            return await self.get_tenant(tenant_id)
        return None

    async def update_tenant_status(
        self, tenant_id: str, status: TenantStatus, reason: str = ""
    ) -> bool:
        """Update tenant status."""
        try:
            tenant = await self.get_tenant(tenant_id)
            if not tenant:
                return False

            old_status = tenant.status
            tenant.status = status
            tenant.updated_at = datetime.now()

            # Audit log
            await self.audit_system.log_event(
                event_type=EventType.SYSTEM_CONFIG_CHANGE,
                action="update_tenant_status",
                outcome="success",
                severity=Severity.MEDIUM,
                details={
                    "tenant_id": tenant_id,
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "reason": reason,
                },
                tenant_id=tenant_id,
            )

            logger.info(
                f"Updated tenant {tenant_id} status from {old_status.value} to {status.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update tenant status: {e}")
            return False

    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a tenant."""
        return await self.update_tenant_status(
            tenant_id, TenantStatus.ACTIVE, "Manual activation"
        )

    async def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Suspend a tenant."""
        return await self.update_tenant_status(
            tenant_id, TenantStatus.SUSPENDED, reason
        )

    async def deactivate_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Deactivate a tenant."""
        return await self.update_tenant_status(
            tenant_id, TenantStatus.DEACTIVATED, reason
        )

    async def update_tenant_tier(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """Update tenant tier and configuration."""
        try:
            tenant = await self.get_tenant(tenant_id)
            if not tenant:
                return False

            old_tier = tenant.tier
            old_config = tenant.configuration

            # Update tier and configuration
            tenant.tier = new_tier
            tenant.configuration = self.tier_configurations[new_tier]
            tenant.updated_at = datetime.now()

            # Audit log
            await self.audit_system.log_event(
                event_type=EventType.SYSTEM_CONFIG_CHANGE,
                action="update_tenant_tier",
                outcome="success",
                severity=Severity.HIGH,
                details={
                    "tenant_id": tenant_id,
                    "old_tier": old_tier.value,
                    "new_tier": new_tier.value,
                    "configuration_changes": {
                        "max_cpu_cores": f"{old_config.max_cpu_cores} -> {tenant.configuration.max_cpu_cores}",
                        "max_memory_gb": f"{old_config.max_memory_gb} -> {tenant.configuration.max_memory_gb}",
                        "max_storage_gb": f"{old_config.max_storage_gb} -> {tenant.configuration.max_storage_gb}",
                    },
                },
                tenant_id=tenant_id,
            )

            logger.info(
                f"Updated tenant {tenant_id} tier from {old_tier.value} to {new_tier.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update tenant tier: {e}")
            return False

    async def update_tenant_configuration(
        self, tenant_id: str, config_updates: dict[str, Any]
    ) -> bool:
        """Update specific tenant configuration settings."""
        try:
            tenant = await self.get_tenant(tenant_id)
            if not tenant:
                return False

            # Update configuration
            for key, value in config_updates.items():
                if hasattr(tenant.configuration, key):
                    setattr(tenant.configuration, key, value)

            tenant.updated_at = datetime.now()

            # Audit log
            await self.audit_system.log_event(
                event_type=EventType.SYSTEM_CONFIG_CHANGE,
                action="update_tenant_configuration",
                outcome="success",
                severity=Severity.MEDIUM,
                details={
                    "tenant_id": tenant_id,
                    "configuration_updates": config_updates,
                },
                tenant_id=tenant_id,
            )

            logger.info(
                f"Updated configuration for tenant {tenant_id}: {config_updates}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update tenant configuration: {e}")
            return False

    async def record_tenant_activity(
        self,
        tenant_id: str,
        activity_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record tenant activity."""
        tenant = await self.get_tenant(tenant_id)
        if tenant:
            tenant.update_last_activity()
            tenant.total_requests += 1

            # Record in telemetry
            self.telemetry.record_detection_metrics(
                duration=0,
                anomaly_count=1,
                algorithm=f"tenant_activity_{activity_type}",
                tenant_id=tenant_id,
            )

    async def check_resource_limits(
        self, tenant_id: str, resource_type: str, amount: float
    ) -> bool:
        """Check if tenant can use requested resources."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        config = tenant.configuration

        # Check specific resource limits
        if resource_type == "cpu_cores" and amount > config.max_cpu_cores:
            return False
        elif resource_type == "memory_gb" and amount > config.max_memory_gb:
            return False
        elif resource_type == "storage_gb" and amount > config.max_storage_gb:
            return False
        elif resource_type == "bandwidth_mbps" and amount > config.max_bandwidth_mbps:
            return False
        elif (
            resource_type == "requests_per_minute"
            and amount > config.max_requests_per_minute
        ):
            return False

        return True

    async def list_tenants(
        self, status: TenantStatus | None = None, tier: TenantTier | None = None
    ) -> list[Tenant]:
        """List tenants with optional filters."""
        tenants = list(self.tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return sorted(tenants, key=lambda t: t.created_at)

    async def get_tenant_usage_summary(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant resource usage summary."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {}

        # This would integrate with actual resource monitoring
        return {
            "tenant_id": tenant_id,
            "status": tenant.status.value,
            "tier": tenant.tier.value,
            "last_activity": (
                tenant.last_activity.isoformat() if tenant.last_activity else None
            ),
            "total_requests": tenant.total_requests,
            "total_data_processed_gb": tenant.total_data_processed_gb,
            "configuration": {
                "max_cpu_cores": tenant.configuration.max_cpu_cores,
                "max_memory_gb": tenant.configuration.max_memory_gb,
                "max_storage_gb": tenant.configuration.max_storage_gb,
                "max_requests_per_minute": tenant.configuration.max_requests_per_minute,
            },
            "features_enabled": list(tenant.configuration.features_enabled),
            "created_at": tenant.created_at.isoformat(),
            "updated_at": tenant.updated_at.isoformat(),
        }

    async def delete_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Delete a tenant (with proper cleanup)."""
        try:
            tenant = await self.get_tenant(tenant_id)
            if not tenant:
                return False

            # Deactivate first
            await self.deactivate_tenant(tenant_id, f"Deletion requested: {reason}")

            # Remove from mappings
            if tenant.api_key in self.tenant_by_api_key:
                del self.tenant_by_api_key[tenant.api_key]

            # Remove tenant
            del self.tenants[tenant_id]

            # Audit log
            await self.audit_system.log_event(
                event_type=EventType.SYSTEM_CONFIG_CHANGE,
                action="delete_tenant",
                outcome="success",
                severity=Severity.HIGH,
                details={
                    "tenant_id": tenant_id,
                    "tenant_name": tenant.name,
                    "reason": reason,
                },
                tenant_id=tenant_id,
            )

            logger.info(f"Deleted tenant {tenant_id} ({tenant.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete tenant: {e}")
            return False

    async def start_monitoring(self) -> None:
        """Start tenant monitoring background task."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started tenant monitoring")

    async def stop_monitoring(self) -> None:
        """Stop tenant monitoring background task."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped tenant monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for tenant health and usage."""
        while self._monitoring_active:
            try:
                await self._check_tenant_health()
                await self._check_resource_usage()
                await self._cleanup_inactive_tenants()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in tenant monitoring loop: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_tenant_health(self) -> None:
        """Check health of all tenants."""
        for tenant in self.tenants.values():
            if tenant.status == TenantStatus.ACTIVE:
                # Check if tenant has been inactive for too long
                if tenant.last_activity:
                    inactive_days = (datetime.now() - tenant.last_activity).days
                    if inactive_days > 30:  # Configurable threshold
                        logger.warning(
                            f"Tenant {tenant.tenant_id} inactive for {inactive_days} days"
                        )

    async def _check_resource_usage(self) -> None:
        """Monitor resource usage across tenants."""
        # This would integrate with actual resource monitoring systems
        for tenant in self.tenants.values():
            if tenant.status == TenantStatus.ACTIVE:
                # Record usage metrics
                self.telemetry.record_system_metrics(
                    {
                        "tenant_requests": tenant.total_requests,
                        "tenant_data_processed": tenant.total_data_processed_gb,
                    }
                )

    async def _cleanup_inactive_tenants(self) -> None:
        """Clean up inactive or expired tenants."""
        cleanup_threshold = datetime.now() - timedelta(days=365)  # 1 year

        for tenant in list(self.tenants.values()):
            if (
                tenant.status == TenantStatus.DEACTIVATED
                and tenant.updated_at < cleanup_threshold
            ):
                logger.info(f"Cleaning up inactive tenant {tenant.tenant_id}")
                # In production, this would trigger a proper deletion workflow

    def get_manager_metrics(self) -> dict[str, Any]:
        """Get tenant manager metrics."""
        status_counts = {}
        tier_counts = {}

        for tenant in self.tenants.values():
            # Count by status
            status = tenant.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count by tier
            tier = tenant.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_tenants": len(self.tenants),
            "status_distribution": status_counts,
            "tier_distribution": tier_counts,
            "monitoring_active": self._monitoring_active,
            "strict_isolation_enabled": self.enable_strict_isolation,
        }


# Global tenant manager instance
_tenant_manager: TenantManager | None = None


def get_tenant_manager(config: Config | None = None) -> TenantManager:
    """Get the global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager(config)
    return _tenant_manager
