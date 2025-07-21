"""
Role-Based Access Control (RBAC) for Pynomaly Detection
=======================================================

Comprehensive RBAC system providing:
- Fine-grained permission management
- Role hierarchy and inheritance
- Resource-level access control
- Dynamic permission evaluation
- Integration with multi-tenancy
"""

import logging
import json
import threading
from typing import Dict, List, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, JSON, Boolean, Text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)

class PermissionType(Enum):
    """Permission type enumeration."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"

class ResourceType(Enum):
    """Resource type enumeration."""
    TENANT = "tenant"
    DATA = "data"
    MODEL = "model"
    DETECTION = "detection"
    REPORT = "report"
    USER = "user"
    ROLE = "role"
    SYSTEM = "system"
    API = "api"

@dataclass
class Permission:
    """Permission definition."""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    permission_type: PermissionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)

@dataclass
class Role:
    """Role definition."""
    role_id: str
    name: str
    description: str
    tenant_id: Optional[str] = None  # None for system-wide roles
    permissions: List[str] = field(default_factory=list)  # Permission IDs
    parent_roles: List[str] = field(default_factory=list)  # Parent role IDs
    is_system_role: bool = False
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class User:
    """User definition for RBAC."""
    user_id: str
    username: str
    email: str
    tenant_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)  # Role IDs
    direct_permissions: List[str] = field(default_factory=list)  # Permission IDs
    is_active: bool = True
    is_system_user: bool = False
    last_login: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessRequest:
    """Access request for evaluation."""
    user_id: str
    resource_type: ResourceType
    permission_type: PermissionType
    resource_id: Optional[str] = None
    tenant_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AccessResult:
    """Result of access evaluation."""
    granted: bool
    reason: str
    evaluated_permissions: List[str] = field(default_factory=list)
    evaluated_roles: List[str] = field(default_factory=list)
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    evaluation_time: float = 0.0

class RoleBasedAccessControl:
    """Comprehensive Role-Based Access Control system."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize RBAC system.
        
        Args:
            database_url: Database connection URL
        """
        self.database_engine = None
        
        # In-memory storage (fallback)
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        
        # Caching
        self.permission_cache: Dict[str, Set[str]] = {}  # user_id -> permissions
        self.role_cache: Dict[str, Set[str]] = {}  # user_id -> roles
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize storage
        if database_url and SQLALCHEMY_AVAILABLE:
            self._initialize_database(database_url)
        
        # Initialize default permissions and roles
        self._initialize_default_permissions()
        self._initialize_default_roles()
        
        logger.info("RBAC system initialized")
    
    def create_permission(self, permission: Permission) -> bool:
        """Create a new permission.
        
        Args:
            permission: Permission to create
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if permission.permission_id in self.permissions:
                    logger.error(f"Permission already exists: {permission.permission_id}")
                    return False
                
                if self.database_engine:
                    self._store_permission_db(permission)
                else:
                    self.permissions[permission.permission_id] = permission
                
                logger.info(f"Permission created: {permission.permission_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create permission {permission.permission_id}: {e}")
            return False
    
    def create_role(self, role: Role) -> bool:
        """Create a new role.
        
        Args:
            role: Role to create
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if role.role_id in self.roles:
                    logger.error(f"Role already exists: {role.role_id}")
                    return False
                
                # Validate parent roles exist
                for parent_id in role.parent_roles:
                    if parent_id not in self.roles:
                        logger.error(f"Parent role not found: {parent_id}")
                        return False
                
                # Validate permissions exist
                for perm_id in role.permissions:
                    if perm_id not in self.permissions:
                        logger.error(f"Permission not found: {perm_id}")
                        return False
                
                if self.database_engine:
                    self._store_role_db(role)
                else:
                    self.roles[role.role_id] = role
                
                # Clear cache for affected users
                self._clear_user_caches_for_role(role.role_id)
                
                logger.info(f"Role created: {role.role_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create role {role.role_id}: {e}")
            return False
    
    def create_user(self, user: User) -> bool:
        """Create a new user.
        
        Args:
            user: User to create
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if user.user_id in self.users:
                    logger.error(f"User already exists: {user.user_id}")
                    return False
                
                # Validate roles exist
                for role_id in user.roles:
                    if role_id not in self.roles:
                        logger.error(f"Role not found: {role_id}")
                        return False
                
                # Validate permissions exist
                for perm_id in user.direct_permissions:
                    if perm_id not in self.permissions:
                        logger.error(f"Permission not found: {perm_id}")
                        return False
                
                if self.database_engine:
                    self._store_user_db(user)
                else:
                    self.users[user.user_id] = user
                
                logger.info(f"User created: {user.user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create user {user.user_id}: {e}")
            return False
    
    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """Assign role to user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                user = self.get_user(user_id)
                if not user:
                    logger.error(f"User not found: {user_id}")
                    return False
                
                if role_id not in self.roles:
                    logger.error(f"Role not found: {role_id}")
                    return False
                
                if role_id not in user.roles:
                    user.roles.append(role_id)
                    user.last_updated = datetime.now()
                    
                    if self.database_engine:
                        self._update_user_db(user)
                    else:
                        self.users[user_id] = user
                    
                    # Clear user cache
                    self._clear_user_cache(user_id)
                    
                    logger.info(f"Role {role_id} assigned to user {user_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to assign role {role_id} to user {user_id}: {e}")
            return False
    
    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Remove role from user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                user = self.get_user(user_id)
                if not user:
                    logger.error(f"User not found: {user_id}")
                    return False
                
                if role_id in user.roles:
                    user.roles.remove(role_id)
                    user.last_updated = datetime.now()
                    
                    if self.database_engine:
                        self._update_user_db(user)
                    else:
                        self.users[user_id] = user
                    
                    # Clear user cache
                    self._clear_user_cache(user_id)
                    
                    logger.info(f"Role {role_id} removed from user {user_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove role {role_id} from user {user_id}: {e}")
            return False
    
    def check_permission(self, access_request: AccessRequest) -> AccessResult:
        """Check if user has permission for requested access.
        
        Args:
            access_request: Access request to evaluate
            
        Returns:
            Access evaluation result
        """
        start_time = datetime.now()
        
        try:
            # Get user
            user = self.get_user(access_request.user_id)
            if not user or not user.is_active:
                return AccessResult(
                    granted=False,
                    reason="User not found or inactive",
                    evaluation_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Get all permissions for user
            user_permissions = self._get_user_permissions(access_request.user_id)
            user_roles = self._get_user_roles(access_request.user_id)
            
            # Evaluate permissions
            granted = False
            matching_permissions = []
            conditions_met = {}
            
            for perm_id in user_permissions:
                permission = self.permissions.get(perm_id)
                if not permission:
                    continue
                
                # Check if permission matches request
                if (permission.resource_type == access_request.resource_type and
                    permission.permission_type == access_request.permission_type):
                    
                    matching_permissions.append(perm_id)
                    
                    # Check conditions
                    conditions_satisfied = self._evaluate_permission_conditions(
                        permission, access_request, user
                    )
                    
                    conditions_met[perm_id] = conditions_satisfied
                    
                    if conditions_satisfied:
                        granted = True
                        break
            
            # Special case for ADMIN permissions
            if not granted:
                admin_permissions = [
                    perm_id for perm_id in user_permissions
                    if perm_id in self.permissions and 
                    self.permissions[perm_id].permission_type == PermissionType.ADMIN
                ]
                
                for perm_id in admin_permissions:
                    permission = self.permissions[perm_id]
                    if permission.resource_type == access_request.resource_type:
                        matching_permissions.append(perm_id)
                        conditions_satisfied = self._evaluate_permission_conditions(
                            permission, access_request, user
                        )
                        conditions_met[perm_id] = conditions_satisfied
                        
                        if conditions_satisfied:
                            granted = True
                            break
            
            reason = "Access granted" if granted else "Insufficient permissions"
            
            result = AccessResult(
                granted=granted,
                reason=reason,
                evaluated_permissions=matching_permissions,
                evaluated_roles=list(user_roles),
                conditions_met=conditions_met,
                evaluation_time=(datetime.now() - start_time).total_seconds()
            )
            
            logger.debug(f"Permission check for {access_request.user_id}: {granted}")
            return result
            
        except Exception as e:
            logger.error(f"Permission check failed for {access_request.user_id}: {e}")
            return AccessResult(
                granted=False,
                reason=f"Evaluation error: {e}",
                evaluation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of permissions
        """
        permission_ids = self._get_user_permissions(user_id)
        return [self.permissions[perm_id] for perm_id in permission_ids if perm_id in self.permissions]
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of roles
        """
        role_ids = self._get_user_roles(user_id)
        return [self.roles[role_id] for role_id in role_ids if role_id in self.roles]
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User or None
        """
        if self.database_engine:
            return self._load_user_db(user_id)
        else:
            return self.users.get(user_id)
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID.
        
        Args:
            role_id: Role identifier
            
        Returns:
            Role or None
        """
        if self.database_engine:
            return self._load_role_db(role_id)
        else:
            return self.roles.get(role_id)
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID.
        
        Args:
            permission_id: Permission identifier
            
        Returns:
            Permission or None
        """
        if self.database_engine:
            return self._load_permission_db(permission_id)
        else:
            return self.permissions.get(permission_id)
    
    def list_users(self, tenant_id: Optional[str] = None) -> List[User]:
        """List users, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            List of users
        """
        if self.database_engine:
            return self._list_users_db(tenant_id)
        else:
            users = list(self.users.values())
            if tenant_id:
                users = [u for u in users if u.tenant_id == tenant_id]
            return users
    
    def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """List roles, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            List of roles
        """
        if self.database_engine:
            return self._list_roles_db(tenant_id)
        else:
            roles = list(self.roles.values())
            if tenant_id:
                roles = [r for r in roles if r.tenant_id == tenant_id or r.is_system_role]
            return roles
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permission IDs for user (including inherited)."""
        # Check cache first
        if user_id in self.permission_cache:
            cache_time = self.cache_timestamps.get(f"perm_{user_id}")
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return self.permission_cache[user_id]
        
        permissions = set()
        user = self.get_user(user_id)
        
        if user:
            # Direct permissions
            permissions.update(user.direct_permissions)
            
            # Role-based permissions
            for role_id in user.roles:
                role_permissions = self._get_role_permissions(role_id)
                permissions.update(role_permissions)
        
        # Cache result
        self.permission_cache[user_id] = permissions
        self.cache_timestamps[f"perm_{user_id}"] = datetime.now()
        
        return permissions
    
    def _get_user_roles(self, user_id: str) -> Set[str]:
        """Get all role IDs for user (including inherited)."""
        # Check cache first
        if user_id in self.role_cache:
            cache_time = self.cache_timestamps.get(f"role_{user_id}")
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return self.role_cache[user_id]
        
        roles = set()
        user = self.get_user(user_id)
        
        if user:
            # Direct roles
            roles.update(user.roles)
            
            # Inherited roles (role hierarchy)
            for role_id in user.roles:
                inherited_roles = self._get_inherited_roles(role_id)
                roles.update(inherited_roles)
        
        # Cache result
        self.role_cache[user_id] = roles
        self.cache_timestamps[f"role_{user_id}"] = datetime.now()
        
        return roles
    
    def _get_role_permissions(self, role_id: str) -> Set[str]:
        """Get all permission IDs for role (including inherited)."""
        permissions = set()
        role = self.get_role(role_id)
        
        if role:
            # Direct permissions
            permissions.update(role.permissions)
            
            # Inherited permissions from parent roles
            for parent_id in role.parent_roles:
                parent_permissions = self._get_role_permissions(parent_id)
                permissions.update(parent_permissions)
        
        return permissions
    
    def _get_inherited_roles(self, role_id: str) -> Set[str]:
        """Get all inherited role IDs."""
        inherited = set()
        role = self.get_role(role_id)
        
        if role:
            for parent_id in role.parent_roles:
                inherited.add(parent_id)
                parent_inherited = self._get_inherited_roles(parent_id)
                inherited.update(parent_inherited)
        
        return inherited
    
    def _evaluate_permission_conditions(self, permission: Permission, 
                                      access_request: AccessRequest, user: User) -> bool:
        """Evaluate permission conditions."""
        if not permission.conditions:
            return True
        
        try:
            for condition_type, condition_value in permission.conditions.items():
                if condition_type == "tenant_match":
                    if condition_value and access_request.tenant_id != user.tenant_id:
                        return False
                
                elif condition_type == "resource_owner":
                    if condition_value and access_request.context.get("owner_id") != user.user_id:
                        return False
                
                elif condition_type == "time_restriction":
                    # Check time-based restrictions
                    current_hour = datetime.now().hour
                    allowed_hours = condition_value.get("allowed_hours", [])
                    if allowed_hours and current_hour not in allowed_hours:
                        return False
                
                elif condition_type == "ip_restriction":
                    # Check IP-based restrictions
                    allowed_ips = condition_value.get("allowed_ips", [])
                    client_ip = access_request.context.get("client_ip")
                    if allowed_ips and client_ip not in allowed_ips:
                        return False
                
                elif condition_type == "custom_function":
                    # Custom condition evaluation
                    function_name = condition_value.get("function")
                    if function_name and hasattr(self, f"_condition_{function_name}"):
                        condition_func = getattr(self, f"_condition_{function_name}")
                        if not condition_func(permission, access_request, user):
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _clear_user_cache(self, user_id: str):
        """Clear cache for specific user."""
        with self.lock:
            if user_id in self.permission_cache:
                del self.permission_cache[user_id]
            if user_id in self.role_cache:
                del self.role_cache[user_id]
            
            # Clear cache timestamps
            perm_key = f"perm_{user_id}"
            role_key = f"role_{user_id}"
            if perm_key in self.cache_timestamps:
                del self.cache_timestamps[perm_key]
            if role_key in self.cache_timestamps:
                del self.cache_timestamps[role_key]
    
    def _clear_user_caches_for_role(self, role_id: str):
        """Clear caches for all users with specific role."""
        affected_users = [
            user_id for user_id, user in self.users.items()
            if role_id in user.roles
        ]
        
        for user_id in affected_users:
            self._clear_user_cache(user_id)
    
    def _initialize_database(self, database_url: str):
        """Initialize database connection and tables."""
        try:
            self.database_engine = create_engine(database_url)
            self._create_rbac_tables()
            logger.info("Database initialized for RBAC")
        except Exception as e:
            logger.error(f"RBAC database initialization failed: {e}")
            self.database_engine = None
    
    def _create_rbac_tables(self):
        """Create database tables for RBAC."""
        metadata = MetaData()
        
        # Permissions table
        permissions_table = Table(
            'rbac_permissions',
            metadata,
            Column('permission_id', String(255), primary_key=True),
            Column('name', String(255)),
            Column('description', Text),
            Column('resource_type', String(50)),
            Column('permission_type', String(50)),
            Column('conditions', JSON),
            Column('created_date', DateTime)
        )
        
        # Roles table
        roles_table = Table(
            'rbac_roles',
            metadata,
            Column('role_id', String(255), primary_key=True),
            Column('name', String(255)),
            Column('description', Text),
            Column('tenant_id', String(255)),
            Column('permissions', JSON),
            Column('parent_roles', JSON),
            Column('is_system_role', Boolean),
            Column('is_active', Boolean),
            Column('created_date', DateTime),
            Column('last_updated', DateTime)
        )
        
        # Users table
        users_table = Table(
            'rbac_users',
            metadata,
            Column('user_id', String(255), primary_key=True),
            Column('username', String(255)),
            Column('email', String(255)),
            Column('tenant_id', String(255)),
            Column('roles', JSON),
            Column('direct_permissions', JSON),
            Column('is_active', Boolean),
            Column('is_system_user', Boolean),
            Column('last_login', DateTime),
            Column('created_date', DateTime),
            Column('metadata', JSON)
        )
        
        metadata.create_all(self.database_engine)
    
    def _initialize_default_permissions(self):
        """Initialize default system permissions."""
        default_permissions = [
            # Data permissions
            Permission("data_read", "Read Data", "Read access to data", ResourceType.DATA, PermissionType.READ),
            Permission("data_write", "Write Data", "Write access to data", ResourceType.DATA, PermissionType.WRITE),
            Permission("data_delete", "Delete Data", "Delete access to data", ResourceType.DATA, PermissionType.DELETE),
            
            # Model permissions
            Permission("model_read", "Read Model", "Read access to models", ResourceType.MODEL, PermissionType.READ),
            Permission("model_write", "Write Model", "Write access to models", ResourceType.MODEL, PermissionType.WRITE),
            Permission("model_execute", "Execute Model", "Execute models", ResourceType.MODEL, PermissionType.EXECUTE),
            Permission("model_delete", "Delete Model", "Delete models", ResourceType.MODEL, PermissionType.DELETE),
            
            # Detection permissions
            Permission("detection_execute", "Execute Detection", "Run anomaly detection", ResourceType.DETECTION, PermissionType.EXECUTE),
            Permission("detection_read", "Read Detection Results", "Read detection results", ResourceType.DETECTION, PermissionType.READ),
            
            # Report permissions
            Permission("report_read", "Read Reports", "Read reports", ResourceType.REPORT, PermissionType.READ),
            Permission("report_write", "Write Reports", "Create/edit reports", ResourceType.REPORT, PermissionType.WRITE),
            
            # User management permissions
            Permission("user_read", "Read Users", "Read user information", ResourceType.USER, PermissionType.READ),
            Permission("user_write", "Write Users", "Create/edit users", ResourceType.USER, PermissionType.WRITE),
            Permission("user_admin", "Admin Users", "Full user administration", ResourceType.USER, PermissionType.ADMIN),
            
            # System permissions
            Permission("system_admin", "System Admin", "Full system administration", ResourceType.SYSTEM, PermissionType.ADMIN),
            Permission("api_access", "API Access", "Access to API endpoints", ResourceType.API, PermissionType.EXECUTE),
        ]
        
        for permission in default_permissions:
            self.permissions[permission.permission_id] = permission
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        default_roles = [
            Role(
                role_id="viewer",
                name="Viewer",
                description="Read-only access to data and results",
                permissions=["data_read", "detection_read", "report_read", "api_access"],
                is_system_role=True
            ),
            Role(
                role_id="analyst",
                name="Analyst",
                description="Data analysis and detection execution",
                permissions=["data_read", "detection_execute", "detection_read", "report_read", "report_write", "api_access"],
                is_system_role=True
            ),
            Role(
                role_id="data_scientist",
                name="Data Scientist",
                description="Model development and advanced analytics",
                permissions=["data_read", "data_write", "model_read", "model_write", "model_execute", 
                           "detection_execute", "detection_read", "report_read", "report_write", "api_access"],
                is_system_role=True
            ),
            Role(
                role_id="admin",
                name="Administrator",
                description="Full administrative access",
                permissions=["data_read", "data_write", "data_delete", "model_read", "model_write", 
                           "model_execute", "model_delete", "detection_execute", "detection_read",
                           "report_read", "report_write", "user_read", "user_write", "user_admin", "api_access"],
                is_system_role=True
            ),
            Role(
                role_id="system_admin",
                name="System Administrator",
                description="Full system administration",
                permissions=["system_admin"],
                is_system_role=True
            )
        ]
        
        for role in default_roles:
            self.roles[role.role_id] = role
    
    # Database operations (simplified implementations)
    def _store_permission_db(self, permission: Permission):
        """Store permission in database."""
        pass
    
    def _store_role_db(self, role: Role):
        """Store role in database."""
        pass
    
    def _store_user_db(self, user: User):
        """Store user in database."""
        pass
    
    def _update_user_db(self, user: User):
        """Update user in database."""
        pass
    
    def _load_permission_db(self, permission_id: str) -> Optional[Permission]:
        """Load permission from database."""
        return None
    
    def _load_role_db(self, role_id: str) -> Optional[Role]:
        """Load role from database."""
        return None
    
    def _load_user_db(self, user_id: str) -> Optional[User]:
        """Load user from database."""
        return None
    
    def _list_users_db(self, tenant_id: Optional[str]) -> List[User]:
        """List users from database."""
        return []
    
    def _list_roles_db(self, tenant_id: Optional[str]) -> List[Role]:
        """List roles from database."""
        return []


class PermissionManager:
    """High-level permission management interface."""
    
    def __init__(self, rbac: RoleBasedAccessControl):
        """Initialize permission manager.
        
        Args:
            rbac: RBAC system instance
        """
        self.rbac = rbac
        logger.info("Permission Manager initialized")
    
    def can_user_access(self, user_id: str, resource_type: str, 
                       permission_type: str, resource_id: Optional[str] = None,
                       tenant_id: Optional[str] = None, **context) -> bool:
        """Simple boolean check for user access.
        
        Args:
            user_id: User identifier
            resource_type: Resource type string
            permission_type: Permission type string
            resource_id: Optional resource identifier
            tenant_id: Optional tenant identifier
            **context: Additional context
            
        Returns:
            True if access granted, False otherwise
        """
        try:
            # Convert string enums
            resource_enum = ResourceType(resource_type.lower())
            permission_enum = PermissionType(permission_type.lower())
            
            request = AccessRequest(
                user_id=user_id,
                resource_type=resource_enum,
                permission_type=permission_enum,
                resource_id=resource_id,
                tenant_id=tenant_id,
                context=context
            )
            
            result = self.rbac.check_permission(request)
            return result.granted
            
        except Exception as e:
            logger.error(f"Access check failed: {e}")
            return False
    
    def require_permission(self, user_id: str, resource_type: str, 
                          permission_type: str, resource_id: Optional[str] = None,
                          tenant_id: Optional[str] = None, **context):
        """Decorator-friendly permission requirement check.
        
        Args:
            user_id: User identifier
            resource_type: Resource type string
            permission_type: Permission type string
            resource_id: Optional resource identifier
            tenant_id: Optional tenant identifier
            **context: Additional context
            
        Raises:
            PermissionError: If access denied
        """
        if not self.can_user_access(user_id, resource_type, permission_type, 
                                   resource_id, tenant_id, **context):
            raise PermissionError(f"Access denied for user {user_id}")
    
    def get_user_accessible_resources(self, user_id: str, resource_type: str,
                                    permission_type: str = "read") -> List[str]:
        """Get list of resources user can access.
        
        Args:
            user_id: User identifier
            resource_type: Resource type string
            permission_type: Permission type string
            
        Returns:
            List of accessible resource IDs
        """
        # This would typically query the resource database
        # and filter based on user permissions
        return []
    
    def create_permission_decorator(self, resource_type: str, permission_type: str):
        """Create a decorator for permission checking.
        
        Args:
            resource_type: Resource type string
            permission_type: Permission type string
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract user_id from function arguments
                user_id = kwargs.get('user_id') or (args[0] if args else None)
                if not user_id:
                    raise ValueError("user_id required for permission check")
                
                # Check permission
                self.require_permission(user_id, resource_type, permission_type)
                
                # Execute function
                return func(*args, **kwargs)
            
            return wrapper
        return decorator