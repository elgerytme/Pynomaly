"""
Multi-Tenant Management Service

Advanced multi-tenant architecture service that provides secure isolation,
resource management, and tenant-specific configurations for anomaly detection.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet

from monorepo.infrastructure.logging.structured_logger import StructuredLogger
from monorepo.infrastructure.monitoring.metrics_service import MetricsService


class TenantStatus(Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


class IsolationLevel(Enum):
    """Data isolation levels."""

    SHARED = "shared"
    TENANT_ISOLATED = "tenant_isolated"
    DEDICATED = "dedicated"
    PRIVATE_CLOUD = "private_cloud"


class ResourceTier(Enum):
    """Resource allocation tiers."""

    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    """Resource quota configuration."""

    cpu_cores: float = 2.0
    memory_gb: float = 4.0
    storage_gb: float = 100.0
    api_requests_per_hour: int = 1000
    concurrent_sessions: int = 10
    max_models: int = 5
    max_datasets: int = 20
    max_users: int = 50
    network_bandwidth_mbps: float = 100.0


@dataclass
class TenantSecurityConfig:
    """Tenant security configuration."""

    encryption_enabled: bool = True
    encryption_key: str | None = None
    audit_logging: bool = True
    access_control_enabled: bool = True
    ip_whitelist: set[str] = field(default_factory=set)
    session_timeout_minutes: int = 60
    password_policy: dict[str, Any] = field(default_factory=dict)
    mfa_required: bool = False
    data_retention_days: int = 365


@dataclass
class TenantConfiguration:
    """Comprehensive tenant configuration."""

    tenant_id: str
    organization_name: str
    contact_email: str
    status: TenantStatus = TenantStatus.PENDING
    isolation_level: IsolationLevel = IsolationLevel.TENANT_ISOLATED
    resource_tier: ResourceTier = ResourceTier.BASIC
    resource_quota: ResourceQuota = field(default_factory=ResourceQuota)
    security_config: TenantSecurityConfig = field(default_factory=TenantSecurityConfig)
    custom_settings: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    billing_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantUsageMetrics:
    """Tenant resource usage metrics."""

    tenant_id: str
    period_start: datetime
    period_end: datetime
    cpu_hours_used: float = 0.0
    memory_gb_hours: float = 0.0
    storage_gb_used: float = 0.0
    api_requests_made: int = 0
    data_processed_gb: float = 0.0
    models_trained: int = 0
    active_sessions: int = 0
    peak_concurrent_users: int = 0
    cost_usd: float = 0.0


@dataclass
class TenantUser:
    """User within a tenant."""

    user_id: str
    tenant_id: str
    username: str
    email: str
    role: str
    permissions: set[str] = field(default_factory=set)
    is_active: bool = True
    last_login: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantResourceManager:
    """Manages resource allocation and monitoring for tenants."""

    def __init__(self):
        self.logger = StructuredLogger("tenant_resource_manager")

        # Resource tier definitions
        self.tier_quotas = {
            ResourceTier.BASIC: ResourceQuota(
                cpu_cores=2.0,
                memory_gb=4.0,
                storage_gb=100.0,
                api_requests_per_hour=1000,
                concurrent_sessions=10,
                max_models=5,
                max_datasets=20,
                max_users=50,
            ),
            ResourceTier.STANDARD: ResourceQuota(
                cpu_cores=4.0,
                memory_gb=8.0,
                storage_gb=500.0,
                api_requests_per_hour=5000,
                concurrent_sessions=25,
                max_models=15,
                max_datasets=100,
                max_users=200,
            ),
            ResourceTier.PREMIUM: ResourceQuota(
                cpu_cores=8.0,
                memory_gb=16.0,
                storage_gb=1000.0,
                api_requests_per_hour=10000,
                concurrent_sessions=50,
                max_models=50,
                max_datasets=500,
                max_users=1000,
            ),
            ResourceTier.ENTERPRISE: ResourceQuota(
                cpu_cores=16.0,
                memory_gb=32.0,
                storage_gb=5000.0,
                api_requests_per_hour=50000,
                concurrent_sessions=100,
                max_models=200,
                max_datasets=2000,
                max_users=5000,
            ),
        }

        # Current usage tracking
        self.usage_tracking: dict[str, TenantUsageMetrics] = {}
        self.resource_locks: dict[str, asyncio.Lock] = {}

    def get_quota_for_tier(self, tier: ResourceTier) -> ResourceQuota:
        """Get resource quota for a tier."""
        return self.tier_quotas[tier]

    async def allocate_resources(
        self, tenant_id: str, resource_request: dict[str, float]
    ) -> bool:
        """Allocate resources to a tenant."""

        # Get or create lock for tenant
        if tenant_id not in self.resource_locks:
            self.resource_locks[tenant_id] = asyncio.Lock()

        async with self.resource_locks[tenant_id]:
            current_usage = self.usage_tracking.get(tenant_id)
            if not current_usage:
                return False

            # Check if allocation would exceed quota
            # Implementation would check actual resource availability

            self.logger.info(
                f"Allocated resources to tenant {tenant_id}: {resource_request}"
            )
            return True

    async def release_resources(
        self, tenant_id: str, resource_release: dict[str, float]
    ):
        """Release resources from a tenant."""

        if tenant_id in self.resource_locks:
            async with self.resource_locks[tenant_id]:
                # Implementation would release actual resources
                self.logger.info(
                    f"Released resources from tenant {tenant_id}: {resource_release}"
                )

    async def check_quota_compliance(
        self, tenant_id: str, quota: ResourceQuota
    ) -> dict[str, bool]:
        """Check if tenant is within quota limits."""

        current_usage = self.usage_tracking.get(tenant_id)
        if not current_usage:
            return {}

        compliance = {
            "cpu_compliant": current_usage.cpu_hours_used <= quota.cpu_cores,
            "memory_compliant": current_usage.memory_gb_hours <= quota.memory_gb,
            "storage_compliant": current_usage.storage_gb_used <= quota.storage_gb,
            "api_compliant": current_usage.api_requests_made
            <= quota.api_requests_per_hour,
            "sessions_compliant": current_usage.active_sessions
            <= quota.concurrent_sessions,
        }

        return compliance

    def update_usage_metrics(self, tenant_id: str, usage_delta: dict[str, float]):
        """Update usage metrics for a tenant."""

        if tenant_id not in self.usage_tracking:
            self.usage_tracking[tenant_id] = TenantUsageMetrics(
                tenant_id=tenant_id,
                period_start=datetime.now(),
                period_end=datetime.now() + timedelta(hours=1),
            )

        usage = self.usage_tracking[tenant_id]

        # Update metrics
        usage.cpu_hours_used += usage_delta.get("cpu_hours", 0.0)
        usage.memory_gb_hours += usage_delta.get("memory_gb_hours", 0.0)
        usage.storage_gb_used = max(
            usage.storage_gb_used, usage_delta.get("storage_gb", 0.0)
        )
        usage.api_requests_made += int(usage_delta.get("api_requests", 0))
        usage.data_processed_gb += usage_delta.get("data_processed_gb", 0.0)
        usage.models_trained += int(usage_delta.get("models_trained", 0))

        # Update concurrent metrics
        if "active_sessions" in usage_delta:
            usage.active_sessions = int(usage_delta["active_sessions"])

        if "concurrent_users" in usage_delta:
            usage.peak_concurrent_users = max(
                usage.peak_concurrent_users, int(usage_delta["concurrent_users"])
            )


class TenantSecurityManager:
    """Manages security for multi-tenant environment."""

    def __init__(self):
        self.logger = StructuredLogger("tenant_security_manager")

        # Encryption keys per tenant
        self.tenant_keys: dict[str, str] = {}

        # Active sessions
        self.active_sessions: dict[str, dict[str, Any]] = {}

        # Audit logs
        self.audit_logs: list[dict[str, Any]] = []

    def generate_tenant_encryption_key(self, tenant_id: str) -> str:
        """Generate encryption key for tenant."""

        key = Fernet.generate_key().decode()
        self.tenant_keys[tenant_id] = key

        self.logger.info(f"Generated encryption key for tenant {tenant_id}")

        return key

    def encrypt_tenant_data(self, tenant_id: str, data: bytes) -> bytes:
        """Encrypt data for a specific tenant."""

        if tenant_id not in self.tenant_keys:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")

        fernet = Fernet(self.tenant_keys[tenant_id].encode())
        return fernet.encrypt(data)

    def decrypt_tenant_data(self, tenant_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt data for a specific tenant."""

        if tenant_id not in self.tenant_keys:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")

        fernet = Fernet(self.tenant_keys[tenant_id].encode())
        return fernet.decrypt(encrypted_data)

    def create_session(self, tenant_id: str, user_id: str, ip_address: str) -> str:
        """Create a new user session."""

        session_id = hashlib.sha256(
            f"{tenant_id}_{user_id}_{time.time()}".encode()
        ).hexdigest()

        self.active_sessions[session_id] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
        }

        self.audit_log(
            "session_created",
            tenant_id,
            user_id,
            {
                "session_id": session_id,
                "ip_address": ip_address,
            },
        )

        return session_id

    def validate_session(
        self, session_id: str, tenant_config: TenantSecurityConfig
    ) -> bool:
        """Validate if session is still valid."""

        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Check session timeout
        session_age = datetime.now() - session["last_activity"]
        if session_age.total_seconds() > tenant_config.session_timeout_minutes * 60:
            self.end_session(session_id)
            return False

        # Update last activity
        session["last_activity"] = datetime.now()

        return True

    def end_session(self, session_id: str):
        """End a user session."""

        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            self.audit_log(
                "session_ended",
                session["tenant_id"],
                session["user_id"],
                {
                    "session_id": session_id,
                    "duration": (
                        datetime.now() - session["created_at"]
                    ).total_seconds(),
                },
            )

            del self.active_sessions[session_id]

    def check_ip_whitelist(
        self, tenant_id: str, ip_address: str, whitelist: set[str]
    ) -> bool:
        """Check if IP address is in tenant's whitelist."""

        if not whitelist:
            return True  # No whitelist means all IPs allowed

        return ip_address in whitelist

    def audit_log(
        self, action: str, tenant_id: str, user_id: str, details: dict[str, Any] = None
    ):
        """Log audit event."""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "details": details or {},
        }

        self.audit_logs.append(log_entry)

        # Keep only recent logs (last 10000)
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]

        self.logger.info(f"Audit log: {action} by {user_id} in tenant {tenant_id}")

    def get_audit_logs(
        self,
        tenant_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get audit logs for a tenant."""

        filtered_logs = [
            log for log in self.audit_logs if log["tenant_id"] == tenant_id
        ]

        # Apply time filters
        if start_time:
            filtered_logs = [
                log
                for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) >= start_time
            ]

        if end_time:
            filtered_logs = [
                log
                for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) <= end_time
            ]

        # Sort by timestamp (newest first) and apply limit
        filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return filtered_logs[:limit]


class MultiTenantManagementService:
    """Comprehensive multi-tenant management service."""

    def __init__(self):
        self.logger = StructuredLogger("multi_tenant_management")
        self.metrics_service = MetricsService()

        # Core components
        self.resource_manager = TenantResourceManager()
        self.security_manager = TenantSecurityManager()

        # Tenant management
        self.tenants: dict[str, TenantConfiguration] = {}
        self.tenant_users: dict[str, list[TenantUser]] = {}

        # Service state
        self.is_running = False

        # Monitoring
        self.monitoring_interval = 300  # 5 minutes

    async def create_tenant(
        self,
        tenant_id: str,
        organization_name: str,
        contact_email: str,
        resource_tier: ResourceTier = ResourceTier.BASIC,
        isolation_level: IsolationLevel = IsolationLevel.TENANT_ISOLATED,
        custom_settings: dict[str, Any] = None,
    ) -> TenantConfiguration:
        """Create a new tenant."""

        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")

        # Get resource quota for tier
        resource_quota = self.resource_manager.get_quota_for_tier(resource_tier)

        # Create security configuration
        security_config = TenantSecurityConfig()

        # Generate encryption key
        encryption_key = self.security_manager.generate_tenant_encryption_key(tenant_id)
        security_config.encryption_key = encryption_key

        # Create tenant configuration
        tenant_config = TenantConfiguration(
            tenant_id=tenant_id,
            organization_name=organization_name,
            contact_email=contact_email,
            resource_tier=resource_tier,
            isolation_level=isolation_level,
            resource_quota=resource_quota,
            security_config=security_config,
            custom_settings=custom_settings or {},
        )

        # Store tenant
        self.tenants[tenant_id] = tenant_config
        self.tenant_users[tenant_id] = []

        # Initialize usage tracking
        self.resource_manager.usage_tracking[tenant_id] = TenantUsageMetrics(
            tenant_id=tenant_id,
            period_start=datetime.now(),
            period_end=datetime.now() + timedelta(days=30),
        )

        # Audit log
        self.security_manager.audit_log(
            "tenant_created",
            tenant_id,
            "system",
            {
                "organization_name": organization_name,
                "resource_tier": resource_tier.value,
                "isolation_level": isolation_level.value,
            },
        )

        self.logger.info(f"Created tenant: {tenant_id} ({organization_name})")

        return tenant_config

    async def update_tenant(
        self,
        tenant_id: str,
        updates: dict[str, Any],
        updated_by: str = "system",
    ) -> TenantConfiguration:
        """Update tenant configuration."""

        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant = self.tenants[tenant_id]

        # Apply updates
        if "organization_name" in updates:
            tenant.organization_name = updates["organization_name"]

        if "contact_email" in updates:
            tenant.contact_email = updates["contact_email"]

        if "status" in updates:
            tenant.status = TenantStatus(updates["status"])

        if "resource_tier" in updates:
            new_tier = ResourceTier(updates["resource_tier"])
            tenant.resource_tier = new_tier
            tenant.resource_quota = self.resource_manager.get_quota_for_tier(new_tier)

        if "custom_settings" in updates:
            tenant.custom_settings.update(updates["custom_settings"])

        # Update timestamp
        tenant.updated_at = datetime.now()

        # Audit log
        self.security_manager.audit_log(
            "tenant_updated", tenant_id, updated_by, updates
        )

        self.logger.info(f"Updated tenant: {tenant_id}")

        return tenant

    async def delete_tenant(self, tenant_id: str, deleted_by: str = "system"):
        """Delete a tenant (archive)."""

        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant = self.tenants[tenant_id]
        tenant.status = TenantStatus.ARCHIVED
        tenant.updated_at = datetime.now()

        # Clean up resources
        await self.resource_manager.release_resources(
            tenant_id,
            {
                "cpu_cores": tenant.resource_quota.cpu_cores,
                "memory_gb": tenant.resource_quota.memory_gb,
            },
        )

        # Audit log
        self.security_manager.audit_log("tenant_deleted", tenant_id, deleted_by)

        self.logger.info(f"Deleted (archived) tenant: {tenant_id}")

    async def create_tenant_user(
        self,
        tenant_id: str,
        username: str,
        email: str,
        role: str,
        permissions: set[str] = None,
        created_by: str = "system",
    ) -> TenantUser:
        """Create a user within a tenant."""

        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Check user limits
        current_users = len(self.tenant_users[tenant_id])
        max_users = self.tenants[tenant_id].resource_quota.max_users

        if current_users >= max_users:
            raise ValueError(
                f"Tenant {tenant_id} has reached maximum user limit ({max_users})"
            )

        # Generate user ID
        user_id = f"{tenant_id}_{username}_{int(time.time())}"

        # Create user
        user = TenantUser(
            user_id=user_id,
            tenant_id=tenant_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions or set(),
        )

        # Add to tenant
        self.tenant_users[tenant_id].append(user)

        # Audit log
        self.security_manager.audit_log(
            "user_created",
            tenant_id,
            created_by,
            {
                "new_user_id": user_id,
                "username": username,
                "role": role,
            },
        )

        self.logger.info(f"Created user {username} in tenant {tenant_id}")

        return user

    async def authenticate_user(
        self,
        tenant_id: str,
        username: str,
        password: str,
        ip_address: str,
    ) -> str | None:
        """Authenticate a user and create session."""

        if tenant_id not in self.tenants:
            return None

        tenant = self.tenants[tenant_id]

        # Check if tenant is active
        if tenant.status != TenantStatus.ACTIVE:
            self.security_manager.audit_log(
                "auth_failed_inactive_tenant", tenant_id, username
            )
            return None

        # Check IP whitelist
        if not self.security_manager.check_ip_whitelist(
            tenant_id, ip_address, tenant.security_config.ip_whitelist
        ):
            self.security_manager.audit_log(
                "auth_failed_ip_blocked",
                tenant_id,
                username,
                {"ip_address": ip_address},
            )
            return None

        # Find user
        user = None
        for tenant_user in self.tenant_users[tenant_id]:
            if tenant_user.username == username and tenant_user.is_active:
                user = tenant_user
                break

        if not user:
            self.security_manager.audit_log(
                "auth_failed_user_not_found", tenant_id, username
            )
            return None

        # In a real implementation, verify password hash
        # For now, we'll assume authentication is successful

        # Create session
        session_id = self.security_manager.create_session(
            tenant_id, user.user_id, ip_address
        )

        # Update last login
        user.last_login = datetime.now()

        self.security_manager.audit_log(
            "user_authenticated",
            tenant_id,
            user.user_id,
            {
                "session_id": session_id,
                "ip_address": ip_address,
            },
        )

        return session_id

    async def authorize_action(
        self,
        session_id: str,
        action: str,
        resource: str = None,
    ) -> bool:
        """Authorize user action."""

        if session_id not in self.security_manager.active_sessions:
            return False

        session = self.security_manager.active_sessions[session_id]
        tenant_id = session["tenant_id"]
        user_id = session["user_id"]

        # Validate session
        tenant = self.tenants[tenant_id]
        if not self.security_manager.validate_session(
            session_id, tenant.security_config
        ):
            return False

        # Find user
        user = None
        for tenant_user in self.tenant_users[tenant_id]:
            if tenant_user.user_id == user_id:
                user = tenant_user
                break

        if not user:
            return False

        # Check permissions
        # In a real implementation, this would be more sophisticated
        authorized = action in user.permissions or "admin" in user.permissions

        # Audit log
        self.security_manager.audit_log(
            "authorization_check",
            tenant_id,
            user_id,
            {
                "action": action,
                "resource": resource,
                "authorized": authorized,
            },
        )

        return authorized

    async def get_tenant_status(self, tenant_id: str) -> dict[str, Any]:
        """Get comprehensive tenant status."""

        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant = self.tenants[tenant_id]
        usage = self.resource_manager.usage_tracking.get(tenant_id)

        # Calculate quota compliance
        compliance = await self.resource_manager.check_quota_compliance(
            tenant_id, tenant.resource_quota
        )

        # Get active sessions
        active_sessions = [
            session
            for session in self.security_manager.active_sessions.values()
            if session["tenant_id"] == tenant_id
        ]

        return {
            "tenant_info": {
                "tenant_id": tenant_id,
                "organization_name": tenant.organization_name,
                "status": tenant.status.value,
                "resource_tier": tenant.resource_tier.value,
                "isolation_level": tenant.isolation_level.value,
                "created_at": tenant.created_at.isoformat(),
                "updated_at": tenant.updated_at.isoformat(),
            },
            "resource_usage": {
                "cpu_hours_used": usage.cpu_hours_used if usage else 0.0,
                "memory_gb_hours": usage.memory_gb_hours if usage else 0.0,
                "storage_gb_used": usage.storage_gb_used if usage else 0.0,
                "api_requests_made": usage.api_requests_made if usage else 0,
                "active_sessions": len(active_sessions),
                "total_users": len(self.tenant_users[tenant_id]),
            },
            "quota_compliance": compliance,
            "security_status": {
                "encryption_enabled": tenant.security_config.encryption_enabled,
                "audit_logging": tenant.security_config.audit_logging,
                "mfa_required": tenant.security_config.mfa_required,
                "session_timeout": tenant.security_config.session_timeout_minutes,
                "ip_whitelist_size": len(tenant.security_config.ip_whitelist),
            },
        }

    async def get_service_statistics(self) -> dict[str, Any]:
        """Get multi-tenant service statistics."""

        total_tenants = len(self.tenants)
        active_tenants = len(
            [t for t in self.tenants.values() if t.status == TenantStatus.ACTIVE]
        )

        # Resource tier distribution
        tier_distribution = {}
        for tier in ResourceTier:
            count = len([t for t in self.tenants.values() if t.resource_tier == tier])
            tier_distribution[tier.value] = count

        # Total usage across all tenants
        total_usage = {
            "cpu_hours": sum(
                u.cpu_hours_used for u in self.resource_manager.usage_tracking.values()
            ),
            "memory_gb_hours": sum(
                u.memory_gb_hours for u in self.resource_manager.usage_tracking.values()
            ),
            "storage_gb": sum(
                u.storage_gb_used for u in self.resource_manager.usage_tracking.values()
            ),
            "api_requests": sum(
                u.api_requests_made
                for u in self.resource_manager.usage_tracking.values()
            ),
        }

        # Active sessions
        total_sessions = len(self.security_manager.active_sessions)

        return {
            "tenant_statistics": {
                "total_tenants": total_tenants,
                "active_tenants": active_tenants,
                "tier_distribution": tier_distribution,
            },
            "resource_usage": total_usage,
            "security_statistics": {
                "active_sessions": total_sessions,
                "audit_logs_count": len(self.security_manager.audit_logs),
            },
            "service_status": {
                "is_running": self.is_running,
                "monitoring_enabled": True,
            },
        }

    async def start_monitoring(self):
        """Start the multi-tenant monitoring system."""

        self.is_running = True
        self.logger.info("Multi-tenant monitoring started")

        while self.is_running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)

    async def _monitoring_cycle(self):
        """Execute monitoring cycle."""

        # Check quota compliance for all tenants
        for tenant_id, tenant in self.tenants.items():
            if tenant.status == TenantStatus.ACTIVE:
                compliance = await self.resource_manager.check_quota_compliance(
                    tenant_id, tenant.resource_quota
                )

                # Alert on quota violations
                for resource, compliant in compliance.items():
                    if not compliant:
                        self.logger.warning(
                            f"Quota violation for tenant {tenant_id}: {resource}"
                        )

                        # Record metric
                        self.metrics_service.record_quota_violation(
                            tenant_id=tenant_id,
                            resource_type=resource,
                        )

        # Clean up expired sessions
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.security_manager.active_sessions.items():
            tenant_id = session["tenant_id"]
            tenant = self.tenants.get(tenant_id)

            if tenant:
                session_age = current_time - session["last_activity"]
                timeout_seconds = tenant.security_config.session_timeout_minutes * 60

                if session_age.total_seconds() > timeout_seconds:
                    expired_sessions.append(session_id)

        # Remove expired sessions
        for session_id in expired_sessions:
            self.security_manager.end_session(session_id)

        self.logger.debug(
            f"Monitoring cycle completed. Cleaned up {len(expired_sessions)} expired sessions"
        )

    async def stop_monitoring(self):
        """Stop the multi-tenant monitoring system."""

        self.is_running = False
        self.logger.info("Multi-tenant monitoring stopped")

    def get_tenant_list(self) -> list[dict[str, Any]]:
        """Get list of all tenants with basic info."""

        tenant_list = []

        for tenant_id, tenant in self.tenants.items():
            usage = self.resource_manager.usage_tracking.get(tenant_id)

            tenant_list.append(
                {
                    "tenant_id": tenant_id,
                    "organization_name": tenant.organization_name,
                    "status": tenant.status.value,
                    "resource_tier": tenant.resource_tier.value,
                    "user_count": len(self.tenant_users.get(tenant_id, [])),
                    "storage_used_gb": usage.storage_gb_used if usage else 0.0,
                    "created_at": tenant.created_at.isoformat(),
                }
            )

        return tenant_list
