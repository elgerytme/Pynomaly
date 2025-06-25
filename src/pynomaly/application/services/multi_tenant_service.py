"""Multi-tenant service for tenant management and resource isolation."""

import asyncio
import secrets
from datetime import datetime
from uuid import UUID

from pynomaly.domain.entities.tenant import (
    ResourceQuotaType,
    SubscriptionTier,
    Tenant,
    TenantStatus,
)


class TenantResourceManager:
    """Manages tenant resource allocation and monitoring."""

    def __init__(self):
        self.resource_locks: dict[UUID, asyncio.Lock] = {}
        self.active_jobs: dict[UUID, set[str]] = {}
        self.resource_usage: dict[UUID, dict[str, float]] = {}

    async def allocate_resources(
        self, tenant_id: UUID, resource_type: str, amount: float
    ) -> bool:
        """Allocate resources for a tenant."""
        async with self._get_tenant_lock(tenant_id):
            if tenant_id not in self.resource_usage:
                self.resource_usage[tenant_id] = {}

            current_usage = self.resource_usage[tenant_id].get(resource_type, 0.0)
            self.resource_usage[tenant_id][resource_type] = current_usage + amount

            return True

    async def deallocate_resources(
        self, tenant_id: UUID, resource_type: str, amount: float
    ) -> bool:
        """Deallocate resources for a tenant."""
        async with self._get_tenant_lock(tenant_id):
            if tenant_id not in self.resource_usage:
                return True

            current_usage = self.resource_usage[tenant_id].get(resource_type, 0.0)
            new_usage = max(0.0, current_usage - amount)
            self.resource_usage[tenant_id][resource_type] = new_usage

            return True

    async def get_resource_usage(self, tenant_id: UUID) -> dict[str, float]:
        """Get current resource usage for a tenant."""
        return self.resource_usage.get(tenant_id, {}).copy()

    async def register_job(self, tenant_id: UUID, job_id: str) -> bool:
        """Register an active job for a tenant."""
        async with self._get_tenant_lock(tenant_id):
            if tenant_id not in self.active_jobs:
                self.active_jobs[tenant_id] = set()

            self.active_jobs[tenant_id].add(job_id)
            return True

    async def unregister_job(self, tenant_id: UUID, job_id: str) -> bool:
        """Unregister a completed job for a tenant."""
        async with self._get_tenant_lock(tenant_id):
            if tenant_id in self.active_jobs:
                self.active_jobs[tenant_id].discard(job_id)
            return True

    async def get_active_job_count(self, tenant_id: UUID) -> int:
        """Get number of active jobs for a tenant."""
        return len(self.active_jobs.get(tenant_id, set()))

    def _get_tenant_lock(self, tenant_id: UUID) -> asyncio.Lock:
        """Get or create a lock for a tenant."""
        if tenant_id not in self.resource_locks:
            self.resource_locks[tenant_id] = asyncio.Lock()
        return self.resource_locks[tenant_id]


class TenantIsolationService:
    """Handles tenant isolation and security."""

    def __init__(self):
        self.encryption_keys: dict[UUID, str] = {}
        self.network_policies: dict[UUID, dict[str, any]] = {}

    async def create_tenant_isolation(self, tenant: Tenant) -> dict[str, any]:
        """Create isolation configuration for a new tenant."""
        # Generate encryption key
        encryption_key = secrets.token_hex(32)
        self.encryption_keys[tenant.tenant_id] = encryption_key

        # Create network isolation policy
        network_policy = {
            "vpc_id": f"vpc-{str(tenant.tenant_id)[:8]}",
            "subnet_id": f"subnet-{str(tenant.tenant_id)[:8]}",
            "security_group_id": f"sg-{str(tenant.tenant_id)[:8]}",
            "allowed_ports": [80, 443, 8080],
            "egress_rules": [
                {"protocol": "tcp", "port": 443, "destination": "0.0.0.0/0"},
                {"protocol": "tcp", "port": 80, "destination": "0.0.0.0/0"},
            ],
            "ingress_rules": [
                {"protocol": "tcp", "port": 443, "source": "0.0.0.0/0"},
                {"protocol": "tcp", "port": 8080, "source": "10.0.0.0/8"},
            ],
        }

        self.network_policies[tenant.tenant_id] = network_policy

        return {
            "encryption_key_id": f"key-{str(tenant.tenant_id)[:8]}",
            "database_schema": tenant.database_schema,
            "network_policy": network_policy,
            "storage_bucket": f"pynomaly-tenant-{str(tenant.tenant_id)[:8]}",
        }

    async def get_tenant_encryption_key(self, tenant_id: UUID) -> str | None:
        """Get encryption key for a tenant."""
        return self.encryption_keys.get(tenant_id)

    async def rotate_encryption_key(self, tenant_id: UUID) -> str:
        """Rotate encryption key for a tenant."""
        new_key = secrets.token_hex(32)
        self.encryption_keys[tenant_id] = new_key
        return f"key-{str(tenant_id)[:8]}-{int(datetime.utcnow().timestamp())}"

    async def validate_tenant_access(self, tenant_id: UUID, resource_path: str) -> bool:
        """Validate tenant access to a resource."""
        # Check if resource path contains tenant identifier
        tenant_identifier = str(tenant_id)[:8]

        # Allow access only to tenant-specific resources
        allowed_patterns = [
            f"/tenant/{tenant_identifier}/",
            f"/api/v1/tenant/{tenant_identifier}/",
            f"/{tenant_identifier}/",
        ]

        return any(pattern in resource_path for pattern in allowed_patterns)


class MultiTenantService:
    """Core multi-tenant service managing tenants and isolation."""

    def __init__(self):
        self.tenants: dict[UUID, Tenant] = {}
        self.tenant_by_name: dict[str, UUID] = {}
        self.resource_manager = TenantResourceManager()
        self.isolation_service = TenantIsolationService()
        self.billing_tracker: dict[UUID, dict[str, any]] = {}

    async def create_tenant(
        self,
        name: str,
        display_name: str,
        contact_email: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE,
        admin_user_id: UUID | None = None,
        description: str = "",
        custom_config: dict[str, any] | None = None,
    ) -> Tenant:
        """Create a new tenant with isolation setup."""

        # Validate tenant name uniqueness
        if name in self.tenant_by_name:
            raise ValueError(f"Tenant with name '{name}' already exists")

        # Create tenant
        tenant = Tenant(
            name=name,
            display_name=display_name,
            contact_email=contact_email,
            subscription_tier=subscription_tier,
            admin_user_id=admin_user_id,
            description=description,
            status=TenantStatus.PENDING_ACTIVATION,
        )

        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(tenant.configuration, key):
                    setattr(tenant.configuration, key, value)

        # Set up isolation
        isolation_config = await self.isolation_service.create_tenant_isolation(tenant)
        tenant.encryption_key_id = isolation_config["encryption_key_id"]
        tenant.network_isolation_config = isolation_config["network_policy"]

        # Store tenant
        self.tenants[tenant.tenant_id] = tenant
        self.tenant_by_name[name] = tenant.tenant_id

        # Initialize billing tracking
        self.billing_tracker[tenant.tenant_id] = {
            "current_period_start": datetime.utcnow(),
            "current_period_charges": 0.0,
            "usage_history": [],
        }

        return tenant

    async def activate_tenant(self, tenant_id: UUID) -> bool:
        """Activate a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.ACTIVE
        tenant.activated_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        return True

    async def suspend_tenant(self, tenant_id: UUID, reason: str = "") -> bool:
        """Suspend a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.SUSPENDED
        tenant.updated_at = datetime.utcnow()
        tenant.metadata["suspension_reason"] = reason
        tenant.metadata["suspended_at"] = datetime.utcnow().isoformat()

        return True

    async def deactivate_tenant(self, tenant_id: UUID) -> bool:
        """Deactivate a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.DEACTIVATED
        tenant.updated_at = datetime.utcnow()

        # Clean up resources
        await self.resource_manager.deallocate_resources(tenant_id, "cpu", 1000.0)
        await self.resource_manager.deallocate_resources(tenant_id, "memory", 1000.0)

        return True

    async def get_tenant(self, tenant_id: UUID) -> Tenant | None:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    async def get_tenant_by_name(self, name: str) -> Tenant | None:
        """Get tenant by name."""
        tenant_id = self.tenant_by_name.get(name)
        if tenant_id:
            return self.tenants.get(tenant_id)
        return None

    async def list_tenants(
        self,
        status: TenantStatus | None = None,
        subscription_tier: SubscriptionTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filtering."""
        tenants = list(self.tenants.values())

        # Apply filters
        if status:
            tenants = [t for t in tenants if t.status == status]

        if subscription_tier:
            tenants = [t for t in tenants if t.subscription_tier == subscription_tier]

        # Sort by creation date (newest first)
        tenants.sort(key=lambda t: t.created_at, reverse=True)

        # Apply pagination
        return tenants[offset : offset + limit]

    async def check_resource_quota(
        self, tenant_id: UUID, quota_type: ResourceQuotaType, amount: int = 1
    ) -> bool:
        """Check if tenant has sufficient quota for resource consumption."""
        tenant = self.tenants.get(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        quota = tenant.get_quota(quota_type)
        if not quota:
            return False

        return quota.can_consume(amount)

    async def consume_resource_quota(
        self, tenant_id: UUID, quota_type: ResourceQuotaType, amount: int = 1
    ) -> bool:
        """Consume tenant resource quota."""
        tenant = self.tenants.get(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        success = tenant.consume_quota(quota_type, amount)

        if success:
            # Track billing
            await self._track_billing_usage(tenant_id, quota_type, amount)

        return success

    async def get_tenant_usage_summary(self, tenant_id: UUID) -> dict[str, any] | None:
        """Get comprehensive usage summary for a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        base_summary = tenant.get_usage_summary()

        # Add real-time resource usage
        resource_usage = await self.resource_manager.get_resource_usage(tenant_id)
        base_summary["real_time_usage"] = resource_usage

        # Add active job count
        active_jobs = await self.resource_manager.get_active_job_count(tenant_id)
        base_summary["active_jobs"] = active_jobs

        # Add billing information
        billing_info = self.billing_tracker.get(tenant_id, {})
        base_summary["billing"] = {
            "current_period_start": (
                billing_info.get("current_period_start", "").isoformat()
                if billing_info.get("current_period_start")
                else ""
            ),
            "current_period_charges": billing_info.get("current_period_charges", 0.0),
            "estimated_monthly_cost": await self._estimate_monthly_cost(tenant_id),
        }

        return base_summary

    async def upgrade_tenant_subscription(
        self, tenant_id: UUID, new_tier: SubscriptionTier
    ) -> bool:
        """Upgrade tenant subscription tier."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        old_tier = tenant.subscription_tier
        tenant.upgrade_subscription(new_tier)

        # Log the upgrade
        tenant.metadata["subscription_history"] = tenant.metadata.get(
            "subscription_history", []
        )
        tenant.metadata["subscription_history"].append(
            {
                "from_tier": old_tier.value,
                "to_tier": new_tier.value,
                "upgraded_at": datetime.utcnow().isoformat(),
                "reason": "user_request",
            }
        )

        return True

    async def reset_tenant_quotas(self, tenant_id: UUID) -> bool:
        """Reset all quotas for a new billing period."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        for quota_type in tenant.resource_quotas:
            tenant.reset_quota(quota_type)

        # Reset billing period
        if tenant_id in self.billing_tracker:
            self.billing_tracker[tenant_id]["current_period_start"] = datetime.utcnow()
            self.billing_tracker[tenant_id]["current_period_charges"] = 0.0

        return True

    async def validate_tenant_access(
        self, tenant_id: UUID, resource_path: str, operation: str = "read"
    ) -> bool:
        """Validate tenant access to a specific resource."""
        tenant = self.tenants.get(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        # Use isolation service for path validation
        path_valid = await self.isolation_service.validate_tenant_access(
            tenant_id, resource_path
        )

        # Additional feature-based validation
        if operation == "gpu_access" and not tenant.configuration.enable_gpu_access:
            return False

        return path_valid

    async def get_tenant_metrics(self, tenant_id: UUID) -> dict[str, any]:
        """Get comprehensive tenant metrics."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}

        return {
            "tenant_info": {
                "tenant_id": str(tenant_id),
                "name": tenant.name,
                "subscription_tier": tenant.subscription_tier.value,
                "status": tenant.status.value,
                "created_at": tenant.created_at.isoformat(),
                "last_activity": (
                    tenant.last_activity.isoformat() if tenant.last_activity else None
                ),
            },
            "resource_usage": await self.resource_manager.get_resource_usage(tenant_id),
            "quota_status": {
                quota_type.value: {
                    "limit": quota.limit if not quota.is_unlimited else "unlimited",
                    "used": quota.used,
                    "usage_percentage": quota.usage_percentage,
                    "is_exceeded": quota.is_exceeded(),
                }
                for quota_type, quota in tenant.resource_quotas.items()
            },
            "active_jobs": await self.resource_manager.get_active_job_count(tenant_id),
            "billing": self.billing_tracker.get(tenant_id, {}),
            "configuration": tenant.configuration.__dict__,
        }

    async def _track_billing_usage(
        self, tenant_id: UUID, quota_type: ResourceQuotaType, amount: int
    ):
        """Track billing usage for a tenant."""
        if tenant_id not in self.billing_tracker:
            return

        # Simple billing rates (would be configurable in production)
        billing_rates = {
            ResourceQuotaType.CPU_HOURS: 0.10,  # $0.10 per CPU hour
            ResourceQuotaType.MEMORY_GB: 0.05,  # $0.05 per GB-hour
            ResourceQuotaType.STORAGE_GB: 0.02,  # $0.02 per GB-month
            ResourceQuotaType.API_REQUESTS: 0.001,  # $0.001 per 1000 requests
        }

        rate = billing_rates.get(quota_type, 0.0)
        charge = amount * rate

        self.billing_tracker[tenant_id]["current_period_charges"] += charge
        self.billing_tracker[tenant_id]["usage_history"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "quota_type": quota_type.value,
                "amount": amount,
                "rate": rate,
                "charge": charge,
            }
        )

    async def _estimate_monthly_cost(self, tenant_id: UUID) -> float:
        """Estimate monthly cost based on current usage patterns."""
        billing_info = self.billing_tracker.get(tenant_id, {})
        current_charges = billing_info.get("current_period_charges", 0.0)
        period_start = billing_info.get("current_period_start")

        if not period_start:
            return 0.0

        # Calculate days elapsed in current period
        days_elapsed = (datetime.utcnow() - period_start).days + 1
        daily_average = current_charges / max(days_elapsed, 1)

        # Estimate monthly cost (30 days)
        return daily_average * 30
