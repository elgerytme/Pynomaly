"""Data isolation service for multi-tenant architecture."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from pynomaly.domain.models.multitenancy import (
    IsolationLevel,
    Tenant,
    TenantContext,
)


class DataIsolationService:
    """Service for ensuring proper data isolation between tenants."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Isolation strategies per tenant
        self.isolation_strategies: dict[UUID, IsolationLevel] = {}

        # Schema mappings for database isolation
        self.tenant_schemas: dict[UUID, str] = {}
        self.tenant_databases: dict[UUID, str] = {}

        # Storage isolation mappings
        self.tenant_storage_buckets: dict[UUID, str] = {}
        self.tenant_storage_prefixes: dict[UUID, str] = {}

        # Network isolation mappings
        self.tenant_namespaces: dict[UUID, str] = {}
        self.tenant_network_policies: dict[UUID, dict[str, Any]] = {}

        # Encryption key management
        self.tenant_encryption_keys: dict[UUID, str] = {}

        # Background tasks
        self.monitoring_tasks: set[asyncio.Task] = set()
        self.is_running = False

        self.logger.info("Data isolation service initialized")

    async def start_monitoring(self) -> None:
        """Start background monitoring for isolation compliance."""

        if self.is_running:
            return

        self.is_running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._isolation_compliance_monitor()),
            asyncio.create_task(self._cross_tenant_access_monitor()),
            asyncio.create_task(self._encryption_compliance_monitor()),
            asyncio.create_task(self._cleanup_isolation_resources()),
        ]

        self.monitoring_tasks.update(tasks)

        self.logger.info("Started data isolation monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""

        self.is_running = False

        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        self.logger.info("Stopped data isolation monitoring")

    async def configure_tenant_isolation(
        self,
        tenant: Tenant,
        isolation_level: IsolationLevel | None = None,
    ) -> None:
        """Configure isolation for a tenant based on their requirements."""

        isolation = isolation_level or tenant.isolation_level
        self.isolation_strategies[tenant.tenant_id] = isolation

        # Configure database isolation
        await self._configure_database_isolation(tenant, isolation)

        # Configure storage isolation
        await self._configure_storage_isolation(tenant, isolation)

        # Configure network isolation
        await self._configure_network_isolation(tenant, isolation)

        # Configure encryption
        await self._configure_encryption(tenant, isolation)

        self.logger.info(f"Configured {isolation.value} isolation for tenant {tenant.name}")

    async def validate_data_access(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        resource_id: str,
        operation: str,
    ) -> bool:
        """Validate that a tenant can access specific data resource."""

        tenant_id = tenant_context.tenant.tenant_id
        isolation = self.isolation_strategies.get(tenant_id)

        if not isolation:
            self.logger.warning(f"No isolation strategy found for tenant {tenant_id}")
            return False

        # Check resource ownership
        if not await self._verify_resource_ownership(tenant_id, resource_type, resource_id):
            self.logger.warning(f"Tenant {tenant_id} attempted unauthorized access to {resource_type}:{resource_id}")
            return False

        # Check operation permissions
        if not await self._verify_operation_permission(tenant_context, operation):
            self.logger.warning(f"Tenant {tenant_id} lacks permission for operation: {operation}")
            return False

        # Isolation-specific validations
        if isolation == IsolationLevel.ISOLATED:
            return await self._validate_isolated_access(tenant_context, resource_type, resource_id)
        elif isolation == IsolationLevel.DEDICATED:
            return await self._validate_dedicated_access(tenant_context, resource_type, resource_id)
        elif isolation == IsolationLevel.SHARED:
            return await self._validate_shared_access(tenant_context, resource_type, resource_id)

        return True

    async def get_tenant_database_config(self, tenant_id: UUID) -> dict[str, str]:
        """Get database configuration for tenant data isolation."""

        isolation = self.isolation_strategies.get(tenant_id)

        if isolation == IsolationLevel.ISOLATED:
            # Completely separate database
            return {
                "database": self.tenant_databases.get(tenant_id, f"pynomaly_tenant_{tenant_id}"),
                "schema": "public",
                "connection_pool": f"tenant_{tenant_id}_pool",
            }

        elif isolation == IsolationLevel.DEDICATED:
            # Dedicated schema in shared database
            return {
                "database": "pynomaly_shared",
                "schema": self.tenant_schemas.get(tenant_id, f"tenant_{tenant_id}"),
                "connection_pool": "shared_pool",
            }

        else:  # SHARED
            # Shared schema with tenant_id prefix
            return {
                "database": "pynomaly_shared",
                "schema": "public",
                "table_prefix": f"tenant_{tenant_id}_",
                "connection_pool": "shared_pool",
            }

    async def get_tenant_storage_config(self, tenant_id: UUID) -> dict[str, str]:
        """Get storage configuration for tenant data isolation."""

        isolation = self.isolation_strategies.get(tenant_id)

        if isolation == IsolationLevel.ISOLATED:
            # Completely separate storage bucket
            return {
                "bucket": self.tenant_storage_buckets.get(tenant_id, f"pynomaly-tenant-{tenant_id}"),
                "prefix": "",
                "encryption_key": self.tenant_encryption_keys.get(tenant_id),
            }

        elif isolation == IsolationLevel.DEDICATED:
            # Dedicated prefix in shared bucket
            return {
                "bucket": "pynomaly-shared-storage",
                "prefix": self.tenant_storage_prefixes.get(tenant_id, f"tenant-{tenant_id}/"),
                "encryption_key": self.tenant_encryption_keys.get(tenant_id),
            }

        else:  # SHARED
            # Shared bucket with tenant ID prefix
            return {
                "bucket": "pynomaly-shared-storage",
                "prefix": f"shared/tenant-{tenant_id}/",
                "encryption_key": "shared_key",
            }

    async def get_tenant_network_config(self, tenant_id: UUID) -> dict[str, Any]:
        """Get network configuration for tenant isolation."""

        isolation = self.isolation_strategies.get(tenant_id)

        if isolation == IsolationLevel.ISOLATED:
            # Completely separate namespace and network policies
            return {
                "namespace": self.tenant_namespaces.get(tenant_id, f"tenant-{tenant_id}"),
                "network_policies": self.tenant_network_policies.get(tenant_id, {}),
                "service_mesh_config": {
                    "istio_enabled": True,
                    "mtls_mode": "STRICT",
                    "isolation_policies": ["DENY_ALL_CROSS_TENANT"],
                },
            }

        elif isolation == IsolationLevel.DEDICATED:
            # Dedicated namespace with shared network
            return {
                "namespace": self.tenant_namespaces.get(tenant_id, f"tenant-{tenant_id}"),
                "network_policies": {
                    "allow_within_namespace": True,
                    "deny_cross_tenant": True,
                },
                "service_mesh_config": {
                    "istio_enabled": True,
                    "mtls_mode": "PERMISSIVE",
                },
            }

        else:  # SHARED
            # Shared namespace with logical separation
            return {
                "namespace": "pynomaly-shared",
                "network_policies": {
                    "tenant_labels_required": True,
                },
                "service_mesh_config": {
                    "istio_enabled": False,
                },
            }

    async def encrypt_tenant_data(
        self,
        tenant_id: UUID,
        data: str | bytes,
    ) -> bytes:
        """Encrypt data using tenant-specific encryption key."""

        encryption_key = self.tenant_encryption_keys.get(tenant_id)
        if not encryption_key:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")

        # In production, this would use proper encryption libraries
        # For now, simulate encryption
        if isinstance(data, str):
            data = data.encode('utf-8')

        # Placeholder encryption (would use actual encryption in production)
        encrypted_data = f"ENCRYPTED_{tenant_id}_{len(data)}_".encode() + data

        return encrypted_data

    async def decrypt_tenant_data(
        self,
        tenant_id: UUID,
        encrypted_data: bytes,
    ) -> bytes:
        """Decrypt data using tenant-specific encryption key."""

        encryption_key = self.tenant_encryption_keys.get(tenant_id)
        if not encryption_key:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")

        # Placeholder decryption (would use actual decryption in production)
        prefix = f"ENCRYPTED_{tenant_id}_".encode()
        if not encrypted_data.startswith(prefix):
            raise ValueError("Invalid encrypted data format")

        # Extract original data (simplified)
        delimiter = b"_"
        parts = encrypted_data.split(delimiter, 3)
        if len(parts) >= 4:
            return parts[3]

        raise ValueError("Failed to decrypt data")

    async def generate_tenant_query_filter(
        self,
        tenant_id: UUID,
        table_name: str,
    ) -> str:
        """Generate SQL filter to ensure tenant data isolation."""

        isolation = self.isolation_strategies.get(tenant_id)

        if isolation == IsolationLevel.ISOLATED:
            # No filter needed - separate database
            return ""

        elif isolation == IsolationLevel.DEDICATED:
            # No filter needed - separate schema
            return ""

        else:  # SHARED
            # Add tenant_id filter
            return f"tenant_id = '{tenant_id}'"

    async def validate_cross_tenant_request(
        self,
        source_tenant_id: UUID,
        target_tenant_id: UUID,
        operation: str,
    ) -> bool:
        """Validate if cross-tenant operation is allowed."""

        if source_tenant_id == target_tenant_id:
            return True

        # Check if either tenant allows cross-tenant operations
        source_isolation = self.isolation_strategies.get(source_tenant_id)
        target_isolation = self.isolation_strategies.get(target_tenant_id)

        # Isolated tenants cannot perform cross-tenant operations
        if (source_isolation == IsolationLevel.ISOLATED or
            target_isolation == IsolationLevel.ISOLATED):
            return False

        # For now, deny all cross-tenant operations
        # In production, this would check specific policies
        return False

    async def audit_data_access(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        resource_id: str,
        operation: str,
        success: bool,
    ) -> None:
        """Audit data access for compliance and security monitoring."""

        audit_data = {
            "tenant_id": str(tenant_context.tenant.tenant_id),
            "user_id": str(tenant_context.user_id) if tenant_context.user_id else None,
            "session_id": tenant_context.session_id,
            "request_id": tenant_context.request_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": tenant_context.ip_address,
            "user_agent": tenant_context.user_agent,
            "isolation_level": tenant_context.tenant.isolation_level.value,
        }

        # Log audit event
        self.logger.info(f"Data access audit: {audit_data}")

        # In production, this would send to audit logging system

    # Private methods for isolation configuration

    async def _configure_database_isolation(
        self,
        tenant: Tenant,
        isolation: IsolationLevel,
    ) -> None:
        """Configure database isolation for tenant."""

        if isolation == IsolationLevel.ISOLATED:
            # Create separate database
            database_name = f"pynomaly_tenant_{tenant.tenant_id}"
            self.tenant_databases[tenant.tenant_id] = database_name

            # In production, this would create the actual database
            self.logger.info(f"Configured isolated database: {database_name}")

        elif isolation == IsolationLevel.DEDICATED:
            # Create dedicated schema
            schema_name = f"tenant_{tenant.tenant_id}"
            self.tenant_schemas[tenant.tenant_id] = schema_name

            # In production, this would create the schema
            self.logger.info(f"Configured dedicated schema: {schema_name}")

        # SHARED uses default configuration

    async def _configure_storage_isolation(
        self,
        tenant: Tenant,
        isolation: IsolationLevel,
    ) -> None:
        """Configure storage isolation for tenant."""

        if isolation == IsolationLevel.ISOLATED:
            # Create separate storage bucket
            bucket_name = f"pynomaly-tenant-{tenant.tenant_id}"
            self.tenant_storage_buckets[tenant.tenant_id] = bucket_name

            # In production, this would create the bucket
            self.logger.info(f"Configured isolated storage bucket: {bucket_name}")

        elif isolation == IsolationLevel.DEDICATED:
            # Create dedicated prefix
            prefix = f"tenant-{tenant.tenant_id}/"
            self.tenant_storage_prefixes[tenant.tenant_id] = prefix

            self.logger.info(f"Configured dedicated storage prefix: {prefix}")

        # SHARED uses default configuration with tenant ID prefix

    async def _configure_network_isolation(
        self,
        tenant: Tenant,
        isolation: IsolationLevel,
    ) -> None:
        """Configure network isolation for tenant."""

        if isolation in [IsolationLevel.ISOLATED, IsolationLevel.DEDICATED]:
            # Create dedicated namespace
            namespace = f"tenant-{tenant.tenant_id}"
            self.tenant_namespaces[tenant.tenant_id] = namespace

            # Configure network policies
            policies = {
                "deny_cross_tenant": True,
                "allow_within_namespace": True,
                "deny_default_namespace": True,
            }

            if isolation == IsolationLevel.ISOLATED:
                policies["deny_all_external"] = True

            self.tenant_network_policies[tenant.tenant_id] = policies

            # In production, this would create Kubernetes namespace and network policies
            self.logger.info(f"Configured network isolation: namespace={namespace}")

        # SHARED uses default namespace with labels

    async def _configure_encryption(
        self,
        tenant: Tenant,
        isolation: IsolationLevel,
    ) -> None:
        """Configure encryption for tenant."""

        if isolation in [IsolationLevel.ISOLATED, IsolationLevel.DEDICATED]:
            # Generate tenant-specific encryption key
            encryption_key = await self._generate_encryption_key(tenant.tenant_id)
            self.tenant_encryption_keys[tenant.tenant_id] = encryption_key

            self.logger.info(f"Configured tenant-specific encryption for {tenant.name}")

        # SHARED uses shared encryption key

    async def _generate_encryption_key(self, tenant_id: UUID) -> str:
        """Generate encryption key for tenant."""

        # In production, this would use proper key management service
        import secrets
        return secrets.token_urlsafe(32)

    # Validation methods

    async def _verify_resource_ownership(
        self,
        tenant_id: UUID,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Verify that tenant owns the specified resource."""

        # In production, this would query the database to verify ownership
        # For now, simulate ownership check
        return True  # Simplified for demo

    async def _verify_operation_permission(
        self,
        tenant_context: TenantContext,
        operation: str,
    ) -> bool:
        """Verify that tenant has permission for operation."""

        # Check if user has required permissions
        required_permissions = {
            "read": ["data.read"],
            "write": ["data.write"],
            "delete": ["data.delete"],
            "admin": ["data.admin"],
        }

        operation_permissions = required_permissions.get(operation, [])

        for permission in operation_permissions:
            if not tenant_context.has_permission(permission):
                return False

        return True

    async def _validate_isolated_access(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Validate access for isolated tenant."""

        # Isolated tenants have complete separation
        # Additional validation can be added here
        return True

    async def _validate_dedicated_access(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Validate access for dedicated tenant."""

        # Dedicated tenants have their own resources but shared infrastructure
        # Additional validation can be added here
        return True

    async def _validate_shared_access(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Validate access for shared tenant."""

        # Shared tenants must have strict validation to prevent cross-tenant access
        tenant_id = tenant_context.tenant.tenant_id

        # Verify resource belongs to tenant (simplified)
        if not resource_id.startswith(f"tenant_{tenant_id}_"):
            return False

        return True

    # Background monitoring tasks

    async def _isolation_compliance_monitor(self) -> None:
        """Monitor isolation compliance across all tenants."""

        while self.is_running:
            try:
                for tenant_id, isolation in self.isolation_strategies.items():
                    await self._check_isolation_compliance(tenant_id, isolation)

            except Exception as e:
                self.logger.error(f"Isolation compliance monitoring error: {e}")

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _check_isolation_compliance(
        self,
        tenant_id: UUID,
        isolation: IsolationLevel,
    ) -> None:
        """Check isolation compliance for specific tenant."""

        # Check database isolation
        if isolation == IsolationLevel.ISOLATED:
            if tenant_id not in self.tenant_databases:
                self.logger.warning(f"Missing isolated database for tenant {tenant_id}")

        elif isolation == IsolationLevel.DEDICATED:
            if tenant_id not in self.tenant_schemas:
                self.logger.warning(f"Missing dedicated schema for tenant {tenant_id}")

        # Check encryption compliance
        if isolation in [IsolationLevel.ISOLATED, IsolationLevel.DEDICATED]:
            if tenant_id not in self.tenant_encryption_keys:
                self.logger.warning(f"Missing encryption key for tenant {tenant_id}")

    async def _cross_tenant_access_monitor(self) -> None:
        """Monitor for unauthorized cross-tenant access attempts."""

        while self.is_running:
            try:
                # This would analyze access logs for cross-tenant violations
                # For now, just log that monitoring is active
                pass

            except Exception as e:
                self.logger.error(f"Cross-tenant access monitoring error: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _encryption_compliance_monitor(self) -> None:
        """Monitor encryption compliance."""

        while self.is_running:
            try:
                # Check that all required data is encrypted
                # Rotate encryption keys if needed
                # Verify encryption key security
                pass

            except Exception as e:
                self.logger.error(f"Encryption compliance monitoring error: {e}")

            await asyncio.sleep(3600)  # Check every hour

    async def _cleanup_isolation_resources(self) -> None:
        """Clean up unused isolation resources."""

        while self.is_running:
            try:
                # Clean up resources for deleted tenants
                # Rotate encryption keys
                # Clean up old network policies
                pass

            except Exception as e:
                self.logger.error(f"Isolation resource cleanup error: {e}")

            await asyncio.sleep(86400)  # Clean up daily

    def get_isolation_summary(self) -> dict[str, Any]:
        """Get summary of isolation configuration."""

        isolation_counts = {}
        for isolation in self.isolation_strategies.values():
            isolation_counts[isolation.value] = isolation_counts.get(isolation.value, 0) + 1

        return {
            "total_tenants": len(self.isolation_strategies),
            "isolation_distribution": isolation_counts,
            "isolated_databases": len(self.tenant_databases),
            "dedicated_schemas": len(self.tenant_schemas),
            "isolated_storage_buckets": len(self.tenant_storage_buckets),
            "dedicated_namespaces": len(self.tenant_namespaces),
            "tenant_encryption_keys": len(self.tenant_encryption_keys),
            "monitoring_active": self.is_running,
        }
