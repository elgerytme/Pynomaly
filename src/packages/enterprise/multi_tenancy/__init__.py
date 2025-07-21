"""Enterprise multi-tenancy support."""

from typing import Optional, List, Dict, Any, Set
from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Tenant:
    """Tenant representation."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    display_name: str = ""
    domain: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass 
class TenantUser:
    """User within a tenant context."""
    user_id: str
    tenant_id: UUID
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


class TenantStorage(ABC):
    """Abstract tenant storage interface."""
    
    @abstractmethod
    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass
    
    @abstractmethod
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create new tenant."""
        pass
    
    @abstractmethod
    async def list_user_tenants(self, user_id: str) -> List[Tenant]:
        """List tenants for user."""
        pass
    
    @abstractmethod
    async def add_user_to_tenant(self, tenant_user: TenantUser) -> None:
        """Add user to tenant."""
        pass


class MultiTenantService:
    """Enterprise multi-tenancy service."""
    
    def __init__(
        self,
        storage: TenantStorage,
        default_tenant_id: Optional[UUID] = None,
        enable_tenant_isolation: bool = True
    ):
        self.storage = storage
        self.default_tenant_id = default_tenant_id
        self.enable_tenant_isolation = enable_tenant_isolation
        self.logger = logger.bind(service="multi_tenant")
    
    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        try:
            tenant = await self.storage.get_tenant(tenant_id)
            if tenant and tenant.is_active:
                return tenant
        except Exception as e:
            self.logger.error("Failed to get tenant", tenant_id=str(tenant_id), error=str(e))
        return None
    
    async def create_tenant(
        self,
        name: str,
        display_name: Optional[str] = None,
        domain: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """Create new tenant."""
        tenant = Tenant(
            name=name,
            display_name=display_name or name,
            domain=domain,
            settings=settings or {},
        )
        
        try:
            created_tenant = await self.storage.create_tenant(tenant)
            self.logger.info("Tenant created", tenant_id=str(created_tenant.id), name=name)
            return created_tenant
        except Exception as e:
            self.logger.error("Failed to create tenant", name=name, error=str(e))
            raise
    
    async def get_user_tenants(self, user_id: str) -> List[Tenant]:
        """Get all tenants for user."""
        try:
            return await self.storage.list_user_tenants(user_id)
        except Exception as e:
            self.logger.error("Failed to get user tenants", user_id=user_id, error=str(e))
            return []
    
    async def add_user_to_tenant(
        self,
        user_id: str,
        tenant_id: UUID,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> None:
        """Add user to tenant with roles/permissions."""
        tenant_user = TenantUser(
            user_id=user_id,
            tenant_id=tenant_id,
            roles=roles or [],
            permissions=permissions or []
        )
        
        try:
            await self.storage.add_user_to_tenant(tenant_user)
            self.logger.info(
                "User added to tenant",
                user_id=user_id,
                tenant_id=str(tenant_id),
                roles=roles
            )
        except Exception as e:
            self.logger.error(
                "Failed to add user to tenant",
                user_id=user_id,
                tenant_id=str(tenant_id),
                error=str(e)
            )
            raise
    
    def get_tenant_context(self, tenant_id: Optional[UUID] = None) -> "TenantContext":
        """Get tenant context for operations."""
        effective_tenant_id = tenant_id or self.default_tenant_id
        return TenantContext(
            tenant_id=effective_tenant_id,
            isolation_enabled=self.enable_tenant_isolation
        )
    
    async def validate_tenant_access(
        self,
        user_id: str,
        tenant_id: UUID,
        required_permission: Optional[str] = None
    ) -> bool:
        """Validate user has access to tenant."""
        if not self.enable_tenant_isolation:
            return True
        
        try:
            user_tenants = await self.get_user_tenants(user_id)
            user_tenant_ids = {t.id for t in user_tenants}
            
            has_access = tenant_id in user_tenant_ids
            
            self.logger.info(
                "Tenant access validation",
                user_id=user_id,
                tenant_id=str(tenant_id),
                has_access=has_access
            )
            
            return has_access
        except Exception as e:
            self.logger.error(
                "Tenant access validation failed",
                user_id=user_id,
                tenant_id=str(tenant_id),
                error=str(e)
            )
            return False


@dataclass
class TenantContext:
    """Tenant context for operations."""
    tenant_id: Optional[UUID]
    isolation_enabled: bool = True
    
    def get_tenant_filter(self) -> Dict[str, Any]:
        """Get filter for tenant-scoped queries."""
        if self.isolation_enabled and self.tenant_id:
            return {"tenant_id": self.tenant_id}
        return {}
    
    def apply_tenant_scope(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tenant scope to query parameters."""
        if self.isolation_enabled and self.tenant_id:
            query_params["tenant_id"] = self.tenant_id
        return query_params


class InMemoryTenantStorage(TenantStorage):
    """In-memory tenant storage for development."""
    
    def __init__(self):
        self.tenants: Dict[UUID, Tenant] = {}
        self.user_tenants: Dict[str, List[UUID]] = {}
        self.tenant_users: Dict[UUID, List[TenantUser]] = {}
    
    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create new tenant."""
        self.tenants[tenant.id] = tenant
        return tenant
    
    async def list_user_tenants(self, user_id: str) -> List[Tenant]:
        """List tenants for user."""
        tenant_ids = self.user_tenants.get(user_id, [])
        return [self.tenants[tid] for tid in tenant_ids if tid in self.tenants]
    
    async def add_user_to_tenant(self, tenant_user: TenantUser) -> None:
        """Add user to tenant."""
        # Add to user->tenants mapping
        if tenant_user.user_id not in self.user_tenants:
            self.user_tenants[tenant_user.user_id] = []
        if tenant_user.tenant_id not in self.user_tenants[tenant_user.user_id]:
            self.user_tenants[tenant_user.user_id].append(tenant_user.tenant_id)
        
        # Add to tenant->users mapping
        if tenant_user.tenant_id not in self.tenant_users:
            self.tenant_users[tenant_user.tenant_id] = []
        self.tenant_users[tenant_user.tenant_id].append(tenant_user)


__all__ = [
    "MultiTenantService",
    "Tenant",
    "TenantUser", 
    "TenantContext",
    "TenantStorage",
    "InMemoryTenantStorage"
]