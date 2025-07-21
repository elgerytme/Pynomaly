"""
Permission and role domain entities for RBAC (Role-Based Access Control).
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class PermissionScope(str, Enum):
    """Permission scope enumeration."""
    GLOBAL = "global"        # System-wide permission
    TENANT = "tenant"        # Tenant-specific permission
    RESOURCE = "resource"    # Specific resource permission


class PermissionAction(str, Enum):
    """Permission action enumeration."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Resource type enumeration."""
    USER = "user"
    TENANT = "tenant"
    ROLE = "role"
    PERMISSION = "permission"
    DATASET = "dataset"
    MODEL = "model"
    EXPERIMENT = "experiment"
    PIPELINE = "pipeline"
    REPORT = "report"
    AUDIT_LOG = "audit_log"
    API_KEY = "api_key"
    INTEGRATION = "integration"
    DASHBOARD = "dashboard"
    ALERT = "alert"
    SETTING = "setting"


class Permission(BaseModel):
    """
    Permission domain entity representing a specific permission in the system.
    
    Permissions are atomic units of access control that can be granted
    to roles and users.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique permission identifier")
    
    # Permission Identification
    name: str = Field(..., min_length=1, max_length=100, description="Permission name")
    code: str = Field(..., min_length=1, max_length=50, description="Unique permission code")
    description: str = Field(..., min_length=1, max_length=500)
    
    # Permission Classification
    scope: PermissionScope = Field(..., description="Permission scope")
    action: PermissionAction = Field(..., description="Action type")
    resource_type: ResourceType = Field(..., description="Resource type")
    
    # Permission Hierarchy
    parent_id: Optional[UUID] = Field(None, description="Parent permission for hierarchy")
    category: str = Field(..., description="Permission category for grouping")
    
    # Constraints
    conditions: Dict[str, any] = Field(default_factory=dict, description="Permission conditions")
    constraints: Dict[str, any] = Field(default_factory=dict, description="Access constraints")
    
    # Status
    is_active: bool = Field(default=True)
    is_system: bool = Field(default=False, description="System-defined permission")
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = Field(None)
    updated_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('code')
    def code_must_be_valid(cls, v):
        """Validate permission code format."""
        if not v.replace('_', '').replace('.', '').replace(':', '').isalnum():
            raise ValueError('Permission code must contain only alphanumeric characters, underscores, dots, and colons')
        return v.upper()
    
    @validator('name')
    def name_must_be_unique_format(cls, v):
        """Validate permission name format."""
        return v.strip()
    
    def matches_resource(self, resource_type: ResourceType, resource_id: Optional[UUID] = None) -> bool:
        """Check if permission matches a specific resource."""
        if self.resource_type != resource_type:
            return False
        
        # If permission has resource-specific constraints
        if self.scope == PermissionScope.RESOURCE and resource_id:
            allowed_resources = self.constraints.get('resource_ids', [])
            return str(resource_id) in allowed_resources
        
        return True
    
    def can_perform_action(self, action: PermissionAction) -> bool:
        """Check if permission allows a specific action."""
        # ADMIN and MANAGE actions include all other actions
        if self.action in [PermissionAction.ADMIN, PermissionAction.MANAGE]:
            return True
        
        return self.action == action
    
    def is_valid_for_scope(self, scope_context: Dict[str, any]) -> bool:
        """Check if permission is valid for the given scope context."""
        if not self.is_active:
            return False
        
        # Evaluate conditions if any
        if self.conditions:
            return self._evaluate_conditions(scope_context)
        
        return True
    
    def _evaluate_conditions(self, context: Dict[str, any]) -> bool:
        """Evaluate permission conditions against context."""
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            elif context_value != condition_value:
                return False
        
        return True
    
    @property
    def full_code(self) -> str:
        """Get full permission code including scope and resource."""
        return f"{self.scope}:{self.resource_type}:{self.action}"
    
    def to_dict(self) -> Dict[str, any]:
        """Convert permission to dictionary."""
        return self.dict()


class Role(BaseModel):
    """
    Role domain entity representing a collection of permissions.
    
    Roles group permissions together for easier management and assignment.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique role identifier")
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID for tenant-specific roles")
    
    # Role Identification
    name: str = Field(..., min_length=1, max_length=100, description="Role name")
    code: str = Field(..., min_length=1, max_length=50, description="Unique role code")
    description: str = Field(..., min_length=1, max_length=500)
    
    # Role Hierarchy
    parent_id: Optional[UUID] = Field(None, description="Parent role for hierarchy")
    level: int = Field(default=0, description="Role hierarchy level")
    
    # Permissions
    permission_ids: Set[UUID] = Field(default_factory=set, description="Assigned permission IDs")
    inherited_permissions: Set[UUID] = Field(default_factory=set, description="Inherited from parent roles")
    
    # Role Properties
    is_default: bool = Field(default=False, description="Default role for new users")
    is_system: bool = Field(default=False, description="System-defined role")
    is_active: bool = Field(default=True)
    
    # Constraints
    max_users: Optional[int] = Field(None, description="Maximum users that can have this role")
    conditions: Dict[str, any] = Field(default_factory=dict, description="Role assignment conditions")
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = Field(None)
    updated_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('code')
    def code_must_be_valid(cls, v):
        """Validate role code format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Role code must contain only alphanumeric characters, underscores, and hyphens')
        return v.upper()
    
    @validator('permission_ids', pre=True)
    def validate_permission_ids(cls, v):
        """Validate permission IDs."""
        if isinstance(v, list):
            return set(v)
        return v or set()
    
    @validator('inherited_permissions', pre=True) 
    def validate_inherited_permissions(cls, v):
        """Validate inherited permission IDs."""
        if isinstance(v, list):
            return set(v)
        return v or set()
    
    def add_permission(self, permission_id: UUID) -> None:
        """Add a permission to the role."""
        self.permission_ids.add(permission_id)
        self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission_id: UUID) -> None:
        """Remove a permission from the role."""
        self.permission_ids.discard(permission_id)
        self.updated_at = datetime.utcnow()
    
    def has_permission(self, permission_id: UUID) -> bool:
        """Check if role has a specific permission."""
        return (
            permission_id in self.permission_ids or 
            permission_id in self.inherited_permissions
        )
    
    def get_all_permissions(self) -> Set[UUID]:
        """Get all permissions (direct and inherited)."""
        return self.permission_ids.union(self.inherited_permissions)
    
    def inherit_from_parent(self, parent_permissions: Set[UUID]) -> None:
        """Inherit permissions from parent role."""
        self.inherited_permissions = parent_permissions
        self.updated_at = datetime.utcnow()
    
    def is_valid_for_tenant(self, tenant_id: UUID) -> bool:
        """Check if role is valid for a specific tenant."""
        # System roles are valid for all tenants
        if self.is_system:
            return True
        
        # Tenant-specific roles
        return self.tenant_id == tenant_id or self.tenant_id is None
    
    @property
    def total_permissions_count(self) -> int:
        """Get total number of permissions (direct + inherited)."""
        return len(self.get_all_permissions())
    
    def to_dict(self) -> Dict[str, any]:
        """Convert role to dictionary."""
        return self.dict()


class RolePermission(BaseModel):
    """
    Role-Permission association entity with additional metadata.
    
    This entity represents the many-to-many relationship between
    roles and permissions with additional context.
    """
    
    id: UUID = Field(default_factory=uuid4)
    role_id: UUID = Field(...)
    permission_id: UUID = Field(...)
    
    # Association Metadata
    granted_by: UUID = Field(..., description="User who granted this permission")
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional Constraints
    conditions: Dict[str, any] = Field(default_factory=dict, description="Specific conditions for this association")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration time")
    
    # Status
    is_active: bool = Field(default=True)
    revoked_at: Optional[datetime] = Field(None)
    revoked_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_valid(self) -> bool:
        """Check if role-permission association is valid."""
        if not self.is_active or self.revoked_at:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def revoke(self, revoked_by: UUID, reason: Optional[str] = None) -> None:
        """Revoke the role-permission association."""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by
        
        if reason:
            self.conditions['revocation_reason'] = reason
    
    def extend_expiration(self, new_expiration: datetime) -> None:
        """Extend the expiration time."""
        self.expires_at = new_expiration


class UserRole(BaseModel):
    """
    User-Role association entity with additional metadata.
    
    This entity represents the assignment of roles to users
    with context and constraints.
    """
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    role_id: UUID = Field(...)
    tenant_id: UUID = Field(...)
    
    # Assignment Metadata
    assigned_by: UUID = Field(..., description="User who assigned this role")
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional Constraints
    conditions: Dict[str, any] = Field(default_factory=dict, description="Role assignment conditions")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration time")
    scope_limitations: Dict[str, any] = Field(default_factory=dict, description="Scope-specific limitations")
    
    # Status
    is_active: bool = Field(default=True)
    revoked_at: Optional[datetime] = Field(None)
    revoked_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_valid(self) -> bool:
        """Check if user-role assignment is valid."""
        if not self.is_active or self.revoked_at:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def revoke(self, revoked_by: UUID, reason: Optional[str] = None) -> None:
        """Revoke the user-role assignment."""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by
        
        if reason:
            self.conditions['revocation_reason'] = reason
    
    def has_scope_limitation(self, scope: str) -> bool:
        """Check if role has limitations in a specific scope."""
        return scope in self.scope_limitations
    
    def get_scope_limitation(self, scope: str) -> any:
        """Get scope limitation value."""
        return self.scope_limitations.get(scope)
    
    def add_scope_limitation(self, scope: str, limitation: any) -> None:
        """Add a scope limitation."""
        self.scope_limitations[scope] = limitation
    
    def remove_scope_limitation(self, scope: str) -> None:
        """Remove a scope limitation."""
        self.scope_limitations.pop(scope, None)


# Pre-defined System Permissions
SYSTEM_PERMISSIONS = {
    # User Management
    "USER.CREATE": Permission(
        name="Create User",
        code="USER.CREATE", 
        description="Create new users",
        scope=PermissionScope.TENANT,
        action=PermissionAction.CREATE,
        resource_type=ResourceType.USER,
        category="User Management",
        is_system=True
    ),
    "USER.READ": Permission(
        name="Read User",
        code="USER.READ",
        description="View user information", 
        scope=PermissionScope.TENANT,
        action=PermissionAction.READ,
        resource_type=ResourceType.USER,
        category="User Management",
        is_system=True
    ),
    "USER.UPDATE": Permission(
        name="Update User",
        code="USER.UPDATE",
        description="Update user information",
        scope=PermissionScope.TENANT,
        action=PermissionAction.UPDATE,
        resource_type=ResourceType.USER,
        category="User Management",
        is_system=True
    ),
    "USER.DELETE": Permission(
        name="Delete User",
        code="USER.DELETE",
        description="Delete users",
        scope=PermissionScope.TENANT,
        action=PermissionAction.DELETE,
        resource_type=ResourceType.USER,
        category="User Management",
        is_system=True
    ),
    "USER.MANAGE": Permission(
        name="Manage Users",
        code="USER.MANAGE",
        description="Full user management capabilities",
        scope=PermissionScope.TENANT,
        action=PermissionAction.MANAGE,
        resource_type=ResourceType.USER,
        category="User Management",
        is_system=True
    ),
    
    # Role Management
    "ROLE.CREATE": Permission(
        name="Create Role",
        code="ROLE.CREATE",
        description="Create new roles",
        scope=PermissionScope.TENANT,
        action=PermissionAction.CREATE,
        resource_type=ResourceType.ROLE,
        category="Role Management",
        is_system=True
    ),
    "ROLE.READ": Permission(
        name="Read Role",
        code="ROLE.READ", 
        description="View role information",
        scope=PermissionScope.TENANT,
        action=PermissionAction.READ,
        resource_type=ResourceType.ROLE,
        category="Role Management",
        is_system=True
    ),
    "ROLE.UPDATE": Permission(
        name="Update Role",
        code="ROLE.UPDATE",
        description="Update role information", 
        scope=PermissionScope.TENANT,
        action=PermissionAction.UPDATE,
        resource_type=ResourceType.ROLE,
        category="Role Management",
        is_system=True
    ),
    "ROLE.DELETE": Permission(
        name="Delete Role",
        code="ROLE.DELETE",
        description="Delete roles",
        scope=PermissionScope.TENANT,
        action=PermissionAction.DELETE,
        resource_type=ResourceType.ROLE,
        category="Role Management",
        is_system=True
    ),
    
    # Tenant Management
    "TENANT.READ": Permission(
        name="Read Tenant",
        code="TENANT.READ",
        description="View tenant information",
        scope=PermissionScope.GLOBAL,
        action=PermissionAction.READ,
        resource_type=ResourceType.TENANT,
        category="Tenant Management",
        is_system=True
    ),
    "TENANT.UPDATE": Permission(
        name="Update Tenant",
        code="TENANT.UPDATE", 
        description="Update tenant settings",
        scope=PermissionScope.TENANT,
        action=PermissionAction.UPDATE,
        resource_type=ResourceType.TENANT,
        category="Tenant Management", 
        is_system=True
    ),
    "TENANT.ADMIN": Permission(
        name="Tenant Admin",
        code="TENANT.ADMIN",
        description="Full tenant administration",
        scope=PermissionScope.TENANT,
        action=PermissionAction.ADMIN,
        resource_type=ResourceType.TENANT,
        category="Tenant Management",
        is_system=True
    ),
    
    # System Administration
    "SYSTEM.ADMIN": Permission(
        name="System Admin",
        code="SYSTEM.ADMIN",
        description="Full system administration",
        scope=PermissionScope.GLOBAL,
        action=PermissionAction.ADMIN,
        resource_type=ResourceType.SETTING,
        category="System Administration",
        is_system=True
    ),
}


# Pre-defined System Roles
SYSTEM_ROLES = {
    "SUPER_ADMIN": Role(
        name="Super Administrator",
        code="SUPER_ADMIN",
        description="Full system access with all permissions",
        level=0,
        is_system=True,
        permission_ids=set(SYSTEM_PERMISSIONS.keys())
    ),
    "TENANT_ADMIN": Role(
        name="Tenant Administrator", 
        code="TENANT_ADMIN",
        description="Full tenant administration capabilities",
        level=1,
        is_system=True,
        permission_ids={
            "USER.MANAGE", "ROLE.CREATE", "ROLE.READ", "ROLE.UPDATE", "ROLE.DELETE",
            "TENANT.READ", "TENANT.UPDATE"
        }
    ),
    "USER_ADMIN": Role(
        name="User Administrator",
        code="USER_ADMIN", 
        description="User management capabilities",
        level=2,
        is_system=True,
        permission_ids={
            "USER.CREATE", "USER.READ", "USER.UPDATE", "USER.DELETE",
            "ROLE.READ"
        }
    ),
    "ANALYST": Role(
        name="Analyst",
        code="ANALYST",
        description="Data analysis and model management",
        level=3,
        is_system=True,
        permission_ids={
            "USER.READ", "ROLE.READ"
        }
    ),
    "VIEWER": Role(
        name="Viewer",
        code="VIEWER",
        description="Read-only access to resources",
        level=4,
        is_system=True,
        is_default=True,
        permission_ids={
            "USER.READ", "ROLE.READ"
        }
    ),
}