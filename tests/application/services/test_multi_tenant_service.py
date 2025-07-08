"""Tests for multi-tenant service."""

from uuid import uuid4

import pytest
from pynomaly.application.services.multi_tenant_service import (
    MultiTenantService,
    TenantIsolationService,
    TenantResourceManager,
)
from pynomaly.domain.entities.tenant import (
    ResourceQuotaType,
    SubscriptionTier,
    Tenant,
    TenantStatus,
)


class TestTenantResourceManager:
    """Test cases for tenant resource manager."""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager instance."""
        return TenantResourceManager()

    @pytest.mark.asyncio
    async def test_allocate_resources(self, resource_manager):
        """Test resource allocation."""
        tenant_id = uuid4()

        success = await resource_manager.allocate_resources(tenant_id, "cpu", 10.0)
        assert success

        usage = await resource_manager.get_resource_usage(tenant_id)
        assert usage["cpu"] == 10.0

        # Allocate more
        await resource_manager.allocate_resources(tenant_id, "cpu", 5.0)
        usage = await resource_manager.get_resource_usage(tenant_id)
        assert usage["cpu"] == 15.0

    @pytest.mark.asyncio
    async def test_deallocate_resources(self, resource_manager):
        """Test resource deallocation."""
        tenant_id = uuid4()

        # Allocate first
        await resource_manager.allocate_resources(tenant_id, "memory", 20.0)

        # Deallocate
        success = await resource_manager.deallocate_resources(tenant_id, "memory", 5.0)
        assert success

        usage = await resource_manager.get_resource_usage(tenant_id)
        assert usage["memory"] == 15.0

        # Deallocate more than available (should not go negative)
        await resource_manager.deallocate_resources(tenant_id, "memory", 30.0)
        usage = await resource_manager.get_resource_usage(tenant_id)
        assert usage["memory"] == 0.0

    @pytest.mark.asyncio
    async def test_job_registration(self, resource_manager):
        """Test job registration and tracking."""
        tenant_id = uuid4()
        job_id = "job_123"

        # Register job
        success = await resource_manager.register_job(tenant_id, job_id)
        assert success

        count = await resource_manager.get_active_job_count(tenant_id)
        assert count == 1

        # Register another job
        await resource_manager.register_job(tenant_id, "job_456")
        count = await resource_manager.get_active_job_count(tenant_id)
        assert count == 2

        # Unregister job
        success = await resource_manager.unregister_job(tenant_id, job_id)
        assert success

        count = await resource_manager.get_active_job_count(tenant_id)
        assert count == 1


class TestTenantIsolationService:
    """Test cases for tenant isolation service."""

    @pytest.fixture
    def isolation_service(self):
        """Create isolation service instance."""
        return TenantIsolationService()

    @pytest.mark.asyncio
    async def test_create_tenant_isolation(self, isolation_service):
        """Test tenant isolation creation."""
        tenant = Tenant(name="test_tenant", contact_email="test@example.com")

        isolation_config = await isolation_service.create_tenant_isolation(tenant)

        assert "encryption_key_id" in isolation_config
        assert "database_schema" in isolation_config
        assert "network_policy" in isolation_config
        assert "storage_bucket" in isolation_config

        # Check network policy structure
        network_policy = isolation_config["network_policy"]
        assert "vpc_id" in network_policy
        assert "subnet_id" in network_policy
        assert "security_group_id" in network_policy
        assert "allowed_ports" in network_policy
        assert "egress_rules" in network_policy
        assert "ingress_rules" in network_policy

    @pytest.mark.asyncio
    async def test_encryption_key_management(self, isolation_service):
        """Test encryption key operations."""
        tenant_id = uuid4()

        # Initially no key
        key = await isolation_service.get_tenant_encryption_key(tenant_id)
        assert key is None

        # Create isolation (which creates key)
        tenant = Tenant(
            tenant_id=tenant_id, name="test", contact_email="test@example.com"
        )
        await isolation_service.create_tenant_isolation(tenant)

        # Now key should exist
        key = await isolation_service.get_tenant_encryption_key(tenant_id)
        assert key is not None

        # Rotate key
        new_key_id = await isolation_service.rotate_encryption_key(tenant_id)
        assert new_key_id is not None
        assert new_key_id.startswith("key-")

    @pytest.mark.asyncio
    async def test_validate_tenant_access(self, isolation_service):
        """Test tenant access validation."""
        tenant_id = uuid4()
        tenant_identifier = str(tenant_id)[:8]

        # Valid paths
        valid_paths = [
            f"/tenant/{tenant_identifier}/data",
            f"/api/v1/tenant/{tenant_identifier}/models",
            f"/{tenant_identifier}/results",
        ]

        for path in valid_paths:
            valid = await isolation_service.validate_tenant_access(tenant_id, path)
            assert valid, f"Path should be valid: {path}"

        # Invalid paths
        invalid_paths = [
            "/tenant/other_tenant/data",
            "/api/v1/admin/users",
            "/global/config",
        ]

        for path in invalid_paths:
            valid = await isolation_service.validate_tenant_access(tenant_id, path)
            assert not valid, f"Path should be invalid: {path}"


class TestMultiTenantService:
    """Test cases for multi-tenant service."""

    @pytest.fixture
    def tenant_service(self):
        """Create multi-tenant service instance."""
        return MultiTenantService()

    @pytest.mark.asyncio
    async def test_create_tenant(self, tenant_service):
        """Test tenant creation."""
        tenant = await tenant_service.create_tenant(
            name="test_tenant",
            display_name="Test Tenant",
            contact_email="test@example.com",
            subscription_tier=SubscriptionTier.BASIC,
            description="Test tenant description",
        )

        assert tenant.name == "test_tenant"
        assert tenant.display_name == "Test Tenant"
        assert tenant.contact_email == "test@example.com"
        assert tenant.subscription_tier == SubscriptionTier.BASIC
        assert tenant.status == TenantStatus.PENDING_ACTIVATION
        assert tenant.database_schema.startswith("tenant_")
        assert tenant.encryption_key_id is not None

        # Check that tenant is stored
        stored_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        assert stored_tenant is not None
        assert stored_tenant.name == "test_tenant"

        # Check that billing tracking is initialized
        assert tenant.tenant_id in tenant_service.billing_tracker

    @pytest.mark.asyncio
    async def test_create_duplicate_tenant_name(self, tenant_service):
        """Test creating tenant with duplicate name fails."""
        await tenant_service.create_tenant(
            name="duplicate_name",
            display_name="First Tenant",
            contact_email="first@example.com",
        )

        with pytest.raises(ValueError, match="already exists"):
            await tenant_service.create_tenant(
                name="duplicate_name",
                display_name="Second Tenant",
                contact_email="second@example.com",
            )

    @pytest.mark.asyncio
    async def test_tenant_lifecycle(self, tenant_service):
        """Test complete tenant lifecycle."""
        # Create tenant
        tenant = await tenant_service.create_tenant(
            name="lifecycle_tenant",
            display_name="Lifecycle Test",
            contact_email="lifecycle@example.com",
        )

        assert tenant.status == TenantStatus.PENDING_ACTIVATION

        # Activate tenant
        success = await tenant_service.activate_tenant(tenant.tenant_id)
        assert success

        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        assert updated_tenant.status == TenantStatus.ACTIVE
        assert updated_tenant.activated_at is not None

        # Suspend tenant
        success = await tenant_service.suspend_tenant(tenant.tenant_id, "Testing")
        assert success

        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        assert updated_tenant.status == TenantStatus.SUSPENDED
        assert updated_tenant.metadata["suspension_reason"] == "Testing"

        # Deactivate tenant
        success = await tenant_service.deactivate_tenant(tenant.tenant_id)
        assert success

        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        assert updated_tenant.status == TenantStatus.DEACTIVATED

    @pytest.mark.asyncio
    async def test_get_tenant_by_name(self, tenant_service):
        """Test getting tenant by name."""
        tenant = await tenant_service.create_tenant(
            name="named_tenant",
            display_name="Named Tenant",
            contact_email="named@example.com",
        )

        retrieved_tenant = await tenant_service.get_tenant_by_name("named_tenant")
        assert retrieved_tenant is not None
        assert retrieved_tenant.tenant_id == tenant.tenant_id

        # Non-existent tenant
        non_existent = await tenant_service.get_tenant_by_name("non_existent")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_list_tenants_with_filters(self, tenant_service):
        """Test listing tenants with various filters."""
        # Create tenants with different statuses and tiers
        await tenant_service.create_tenant(
            name="tenant1",
            display_name="Tenant 1",
            contact_email="1@example.com",
            subscription_tier=SubscriptionTier.FREE,
        )
        tenant2 = await tenant_service.create_tenant(
            name="tenant2",
            display_name="Tenant 2",
            contact_email="2@example.com",
            subscription_tier=SubscriptionTier.BASIC,
        )
        await tenant_service.create_tenant(
            name="tenant3",
            display_name="Tenant 3",
            contact_email="3@example.com",
            subscription_tier=SubscriptionTier.FREE,
        )

        # Activate one tenant
        await tenant_service.activate_tenant(tenant2.tenant_id)

        # List all tenants
        all_tenants = await tenant_service.list_tenants()
        assert len(all_tenants) >= 3

        # Filter by status
        pending_tenants = await tenant_service.list_tenants(
            status=TenantStatus.PENDING_ACTIVATION
        )
        assert len(pending_tenants) >= 2

        active_tenants = await tenant_service.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active_tenants) >= 1

        # Filter by subscription tier
        free_tenants = await tenant_service.list_tenants(
            subscription_tier=SubscriptionTier.FREE
        )
        assert len(free_tenants) >= 2

        basic_tenants = await tenant_service.list_tenants(
            subscription_tier=SubscriptionTier.BASIC
        )
        assert len(basic_tenants) >= 1

        # Test pagination
        limited_tenants = await tenant_service.list_tenants(limit=2)
        assert len(limited_tenants) <= 2

    @pytest.mark.asyncio
    async def test_resource_quota_management(self, tenant_service):
        """Test resource quota checking and consumption."""
        tenant = await tenant_service.create_tenant(
            name="quota_tenant",
            display_name="Quota Test",
            contact_email="quota@example.com",
            subscription_tier=SubscriptionTier.BASIC,
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        # Check quota availability
        can_consume = await tenant_service.check_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 100
        )
        assert can_consume

        # Consume quota
        success = await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 100
        )
        assert success

        # Check updated tenant state
        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        api_quota = updated_tenant.get_quota(ResourceQuotaType.API_REQUESTS)
        assert api_quota.used == 100

        # Try to consume more than available (should fail for limited quotas)
        very_large_amount = 50000  # Exceeds BASIC tier limit
        can_consume = await tenant_service.check_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, very_large_amount
        )
        assert not can_consume

        success = await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, very_large_amount
        )
        assert not success

    @pytest.mark.asyncio
    async def test_subscription_upgrade(self, tenant_service):
        """Test subscription tier upgrade."""
        tenant = await tenant_service.create_tenant(
            name="upgrade_tenant",
            display_name="Upgrade Test",
            contact_email="upgrade@example.com",
            subscription_tier=SubscriptionTier.FREE,
        )

        # Upgrade to PROFESSIONAL
        success = await tenant_service.upgrade_tenant_subscription(
            tenant.tenant_id, SubscriptionTier.PROFESSIONAL
        )
        assert success

        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        assert updated_tenant.subscription_tier == SubscriptionTier.PROFESSIONAL

        # Check that quotas were updated
        cpu_quota = updated_tenant.get_quota(ResourceQuotaType.CPU_HOURS)
        assert cpu_quota.limit == 500  # PROFESSIONAL tier limit

        # Check metadata for upgrade history
        assert "subscription_history" in updated_tenant.metadata
        history = updated_tenant.metadata["subscription_history"]
        assert len(history) == 1
        assert history[0]["from_tier"] == "free"
        assert history[0]["to_tier"] == "professional"

    @pytest.mark.asyncio
    async def test_quota_reset(self, tenant_service):
        """Test quota reset for new billing period."""
        tenant = await tenant_service.create_tenant(
            name="reset_tenant",
            display_name="Reset Test",
            contact_email="reset@example.com",
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        # Consume some quota
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 500
        )

        # Verify quota is consumed
        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        api_quota = updated_tenant.get_quota(ResourceQuotaType.API_REQUESTS)
        assert api_quota.used == 500

        # Reset quotas
        success = await tenant_service.reset_tenant_quotas(tenant.tenant_id)
        assert success

        # Verify quotas are reset
        updated_tenant = await tenant_service.get_tenant(tenant.tenant_id)
        api_quota = updated_tenant.get_quota(ResourceQuotaType.API_REQUESTS)
        assert api_quota.used == 0

        # Verify billing period is reset
        billing_info = tenant_service.billing_tracker[tenant.tenant_id]
        assert billing_info["current_period_charges"] == 0.0

    @pytest.mark.asyncio
    async def test_tenant_usage_summary(self, tenant_service):
        """Test getting tenant usage summary."""
        tenant = await tenant_service.create_tenant(
            name="summary_tenant",
            display_name="Summary Test",
            contact_email="summary@example.com",
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        # Consume some resources
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 100
        )
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.CPU_HOURS, 5
        )

        # Add some real-time resource usage
        await tenant_service.resource_manager.allocate_resources(
            tenant.tenant_id, "memory", 1024.0
        )

        # Register an active job
        await tenant_service.resource_manager.register_job(
            tenant.tenant_id, "test_job_123"
        )

        # Get usage summary
        summary = await tenant_service.get_tenant_usage_summary(tenant.tenant_id)

        assert summary is not None
        assert summary["tenant_id"] == str(tenant.tenant_id)
        assert summary["name"] == "summary_tenant"
        assert summary["subscription_tier"] == "free"
        assert summary["status"] == "active"

        # Check quotas
        assert "quotas" in summary
        api_quota = summary["quotas"]["api_requests"]
        assert api_quota["used"] == 100

        # Check real-time usage
        assert "real_time_usage" in summary
        assert summary["real_time_usage"]["memory"] == 1024.0

        # Check active jobs
        assert summary["active_jobs"] == 1

        # Check billing info
        assert "billing" in summary
        billing = summary["billing"]
        assert "current_period_start" in billing
        assert "current_period_charges" in billing
        assert "estimated_monthly_cost" in billing

    @pytest.mark.asyncio
    async def test_validate_tenant_access(self, tenant_service):
        """Test tenant access validation."""
        tenant = await tenant_service.create_tenant(
            name="access_tenant",
            display_name="Access Test",
            contact_email="access@example.com",
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        tenant_identifier = str(tenant.tenant_id)[:8]

        # Valid access
        valid = await tenant_service.validate_tenant_access(
            tenant.tenant_id, f"/tenant/{tenant_identifier}/data"
        )
        assert valid

        # Invalid access
        invalid = await tenant_service.validate_tenant_access(
            tenant.tenant_id, "/tenant/other_tenant/data"
        )
        assert not invalid

        # Inactive tenant should be denied
        await tenant_service.suspend_tenant(tenant.tenant_id)
        suspended_access = await tenant_service.validate_tenant_access(
            tenant.tenant_id, f"/tenant/{tenant_identifier}/data"
        )
        assert not suspended_access

    @pytest.mark.asyncio
    async def test_tenant_metrics(self, tenant_service):
        """Test getting comprehensive tenant metrics."""
        tenant = await tenant_service.create_tenant(
            name="metrics_tenant",
            display_name="Metrics Test",
            contact_email="metrics@example.com",
            subscription_tier=SubscriptionTier.BASIC,
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        # Add some usage
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 250
        )
        await tenant_service.resource_manager.allocate_resources(
            tenant.tenant_id, "cpu", 4.0
        )
        await tenant_service.resource_manager.register_job(tenant.tenant_id, "job1")
        await tenant_service.resource_manager.register_job(tenant.tenant_id, "job2")

        metrics = await tenant_service.get_tenant_metrics(tenant.tenant_id)

        assert metrics is not None

        # Check tenant info
        tenant_info = metrics["tenant_info"]
        assert tenant_info["name"] == "metrics_tenant"
        assert tenant_info["subscription_tier"] == "basic"
        assert tenant_info["status"] == "active"

        # Check resource usage
        resource_usage = metrics["resource_usage"]
        assert resource_usage["cpu"] == 4.0

        # Check quota status
        quota_status = metrics["quota_status"]
        api_quota = quota_status["api_requests"]
        assert api_quota["used"] == 250
        assert api_quota["limit"] == 10000  # BASIC tier limit
        assert api_quota["usage_percentage"] == 2.5

        # Check active jobs
        assert metrics["active_jobs"] == 2

        # Check billing
        billing = metrics["billing"]
        assert "current_period_start" in billing
        assert "current_period_charges" in billing

        # Check configuration
        config = metrics["configuration"]
        assert "max_concurrent_jobs" in config
        assert "enable_gpu_access" in config

    @pytest.mark.asyncio
    async def test_billing_tracking(self, tenant_service):
        """Test billing usage tracking."""
        tenant = await tenant_service.create_tenant(
            name="billing_tenant",
            display_name="Billing Test",
            contact_email="billing@example.com",
        )

        await tenant_service.activate_tenant(tenant.tenant_id)

        # Consume various resources
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.API_REQUESTS, 1000
        )
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.CPU_HOURS, 10
        )
        await tenant_service.consume_resource_quota(
            tenant.tenant_id, ResourceQuotaType.MEMORY_GB, 20
        )

        # Check billing tracker
        billing_info = tenant_service.billing_tracker[tenant.tenant_id]

        assert billing_info["current_period_charges"] > 0
        assert len(billing_info["usage_history"]) == 3

        # Check usage history details
        history = billing_info["usage_history"]

        # Find API requests entry
        api_entry = next(h for h in history if h["quota_type"] == "api_requests")
        assert api_entry["amount"] == 1000
        assert api_entry["charge"] == 1.0  # 1000 * 0.001

        # Find CPU hours entry
        cpu_entry = next(h for h in history if h["quota_type"] == "cpu_hours")
        assert cpu_entry["amount"] == 10
        assert cpu_entry["charge"] == 1.0  # 10 * 0.10

        # Find memory entry
        memory_entry = next(h for h in history if h["quota_type"] == "memory_gb")
        assert memory_entry["amount"] == 20
        assert memory_entry["charge"] == 1.0  # 20 * 0.05

    @pytest.mark.asyncio
    async def test_nonexistent_tenant_operations(self, tenant_service):
        """Test operations on non-existent tenants."""
        fake_tenant_id = uuid4()

        # Get non-existent tenant
        tenant = await tenant_service.get_tenant(fake_tenant_id)
        assert tenant is None

        # Activate non-existent tenant
        success = await tenant_service.activate_tenant(fake_tenant_id)
        assert not success

        # Check quota for non-existent tenant
        can_consume = await tenant_service.check_resource_quota(
            fake_tenant_id, ResourceQuotaType.API_REQUESTS, 1
        )
        assert not can_consume

        # Consume quota for non-existent tenant
        success = await tenant_service.consume_resource_quota(
            fake_tenant_id, ResourceQuotaType.API_REQUESTS, 1
        )
        assert not success

        # Get usage summary for non-existent tenant
        summary = await tenant_service.get_tenant_usage_summary(fake_tenant_id)
        assert summary is None

        # Validate access for non-existent tenant
        valid = await tenant_service.validate_tenant_access(
            fake_tenant_id, "/some/path"
        )
        assert not valid
