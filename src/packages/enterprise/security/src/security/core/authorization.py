"""Enterprise authorization and access control management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import structlog

from ..config.security_config import SecurityConfig
from ...shared.infrastructure.exceptions.base_exceptions import (
    BaseApplicationError,
    ErrorCategory,
    ErrorSeverity
)
from ...shared.infrastructure.logging.structured_logging import StructuredLogger


logger = structlog.get_logger()


class AuthorizationError(BaseApplicationError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class PermissionType(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(Enum):
    """Resource types for access control."""
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    DATA = "data"
    API = "api"
    SYSTEM = "system"
    REPORT = "report"
    AUDIT = "audit"


@dataclass
class Permission:
    """Permission definition."""
    name: str
    description: str
    resource_type: ResourceType
    permission_type: PermissionType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self) -> str:
        return f"{self.resource_type.value}:{self.permission_type.value}:{self.name}"


@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessPolicy:
    """Access control policy."""
    name: str
    description: str
    resource_pattern: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    effect: str = "allow"  # allow or deny
    priority: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessRequest:
    """Access request context."""
    user_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessResult:
    """Access control decision result."""
    allowed: bool
    reason: str
    matched_policies: List[str] = field(default_factory=list)
    user_roles: List[str] = field(default_factory=list)
    user_permissions: List[str] = field(default_factory=list)


class PolicyEvaluator(ABC):
    """Abstract base class for policy evaluators."""
    
    @abstractmethod
    def evaluate(self, request: AccessRequest, context: Dict[str, Any]) -> bool:
        """Evaluate access request against policy."""
        pass


class AttributeBasedPolicyEvaluator(PolicyEvaluator):
    """Attribute-based access control (ABAC) policy evaluator."""
    
    def __init__(self, policy: AccessPolicy):
        self.policy = policy
    
    def evaluate(self, request: AccessRequest, context: Dict[str, Any]) -> bool:
        """Evaluate request using attribute-based rules."""
        try:
            # Check resource pattern match
            if not self._matches_pattern(request.resource, self.policy.resource_pattern):
                return False
            
            # Evaluate conditions
            for condition_key, condition_value in self.policy.conditions.items():
                if not self._evaluate_condition(condition_key, condition_value, request, context):
                    return False
            
            return self.policy.effect == "allow"
            
        except Exception as e:
            logger.error("Policy evaluation failed", policy=self.policy.name, error=str(e))
            return False
    
    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(resource, pattern)
    
    def _evaluate_condition(
        self, 
        key: str, 
        value: Any, 
        request: AccessRequest, 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate individual condition."""
        if key == "time_range":
            return self._evaluate_time_condition(value, request.timestamp)
        elif key == "ip_address":
            return self._evaluate_ip_condition(value, context.get("ip_address"))
        elif key == "user_attribute":
            return self._evaluate_user_attribute_condition(value, context.get("user"))
        elif key == "resource_attribute":
            return self._evaluate_resource_attribute_condition(value, request.resource)
        else:
            return True
    
    def _evaluate_time_condition(self, time_range: Dict[str, str], timestamp: datetime) -> bool:
        """Evaluate time-based condition."""
        start_time = datetime.fromisoformat(time_range.get("start", "00:00:00"))
        end_time = datetime.fromisoformat(time_range.get("end", "23:59:59"))
        current_time = timestamp.time()
        return start_time.time() <= current_time <= end_time.time()
    
    def _evaluate_ip_condition(self, allowed_ips: List[str], user_ip: str) -> bool:
        """Evaluate IP address condition."""
        if not user_ip or not allowed_ips:
            return True
        return user_ip in allowed_ips
    
    def _evaluate_user_attribute_condition(self, condition: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Evaluate user attribute condition."""
        if not user:
            return False
        
        for attr_key, attr_value in condition.items():
            user_value = user.get(attr_key)
            if user_value != attr_value:
                return False
        return True
    
    def _evaluate_resource_attribute_condition(self, condition: Dict[str, Any], resource: str) -> bool:
        """Evaluate resource attribute condition."""
        # Simple implementation - can be extended for complex resource attributes
        return True


class AuthorizationManager:
    """Enterprise role-based and attribute-based access control system.
    
    Provides comprehensive authorization capabilities including:
    - Role-based access control (RBAC)
    - Attribute-based access control (ABAC)
    - Permission management
    - Access policy evaluation
    - Audit logging
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = StructuredLogger(config.logging)
        
        # In-memory stores (in production, use external stores)
        self._permissions: Dict[str, Permission] = {}
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._policies: Dict[str, AccessPolicy] = {}
        self._policy_evaluators: Dict[str, PolicyEvaluator] = {}
        
        # Initialize default permissions and roles
        self._initialize_default_permissions()
        self._initialize_default_roles()
    
    def create_permission(
        self, 
        name: str, 
        description: str, 
        resource_type: ResourceType, 
        permission_type: PermissionType
    ) -> Permission:
        """Create a new permission."""
        try:
            permission = Permission(
                name=name,
                description=description,
                resource_type=resource_type,
                permission_type=permission_type
            )
            
            permission_key = str(permission)
            self._permissions[permission_key] = permission
            
            self.logger.info(
                "Permission created",
                permission=permission_key,
                resource_type=resource_type.value,
                permission_type=permission_type.value
            )
            
            return permission
            
        except Exception as e:
            self.logger.error("Permission creation failed", error=str(e))
            raise AuthorizationError("Permission creation failed") from e
    
    def create_role(self, name: str, description: str, permissions: List[str] = None) -> Role:
        """Create a new role with permissions."""
        try:
            role = Role(
                name=name,
                description=description,
                permissions=set(permissions or [])
            )
            
            self._roles[name] = role
            
            self.logger.info(
                "Role created",
                role=name,
                permissions_count=len(role.permissions)
            )
            
            return role
            
        except Exception as e:
            self.logger.error("Role creation failed", error=str(e))
            raise AuthorizationError("Role creation failed") from e
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        try:
            if role_name not in self._roles:
                return False
            
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()
            
            self._user_roles[user_id].add(role_name)
            
            self.logger.info(
                "Role assigned to user",
                user_id=user_id,
                role=role_name
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Role assignment failed", error=str(e))
            return False
    
    def revoke_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        try:
            if user_id not in self._user_roles:
                return False
            
            self._user_roles[user_id].discard(role_name)
            
            self.logger.info(
                "Role revoked from user",
                user_id=user_id,
                role=role_name
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Role revocation failed", error=str(e))
            return False
    
    def create_access_policy(
        self, 
        name: str, 
        description: str, 
        resource_pattern: str,
        conditions: Dict[str, Any] = None,
        effect: str = "allow",
        priority: int = 100
    ) -> AccessPolicy:
        """Create an access control policy."""
        try:
            policy = AccessPolicy(
                name=name,
                description=description,
                resource_pattern=resource_pattern,
                conditions=conditions or {},
                effect=effect,
                priority=priority
            )
            
            self._policies[name] = policy
            self._policy_evaluators[name] = AttributeBasedPolicyEvaluator(policy)
            
            self.logger.info(
                "Access policy created",
                policy=name,
                resource_pattern=resource_pattern,
                effect=effect
            )
            
            return policy
            
        except Exception as e:
            self.logger.error("Policy creation failed", error=str(e))
            raise AuthorizationError("Policy creation failed") from e
    
    def check_permission(
        self, 
        user_id: str, 
        resource: str, 
        action: str,
        context: Dict[str, Any] = None
    ) -> AccessResult:
        """Check if user has permission for resource and action."""
        try:
            request = AccessRequest(
                user_id=user_id,
                resource=resource,
                action=action,
                context=context or {}
            )
            
            # Get user roles and permissions
            user_roles = list(self._user_roles.get(user_id, set()))
            user_permissions = self._get_user_permissions(user_id)
            
            # Check direct permission match
            required_permission = f"{resource}:{action}"
            if required_permission in user_permissions:
                result = AccessResult(
                    allowed=True,
                    reason="Direct permission match",
                    user_roles=user_roles,
                    user_permissions=user_permissions
                )
                
                self._log_access_decision(request, result)
                return result
            
            # Evaluate policies
            matched_policies = []
            policy_results = []
            
            for policy_name, policy in self._policies.items():
                evaluator = self._policy_evaluators[policy_name]
                if evaluator.evaluate(request, context or {}):
                    matched_policies.append(policy_name)
                    policy_results.append((policy.priority, policy.effect == "allow"))
            
            # Determine final decision based on policies
            if matched_policies:
                # Sort by priority (higher priority first)
                policy_results.sort(key=lambda x: x[0], reverse=True)
                allowed = policy_results[0][1]  # Use highest priority policy result
                
                result = AccessResult(
                    allowed=allowed,
                    reason=f"Policy-based decision: {matched_policies[0]}",
                    matched_policies=matched_policies,
                    user_roles=user_roles,
                    user_permissions=user_permissions
                )
            else:
                # Default deny
                result = AccessResult(
                    allowed=False,
                    reason="No matching permissions or policies",
                    user_roles=user_roles,
                    user_permissions=user_permissions
                )
            
            self._log_access_decision(request, result)
            return result
            
        except Exception as e:
            self.logger.error("Permission check failed", error=str(e))
            return AccessResult(
                allowed=False,
                reason="Permission check failed due to error"
            )
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to a user."""
        return list(self._user_roles.get(user_id, set()))
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user."""
        return self._get_user_permissions(user_id)
    
    def get_role_permissions(self, role_name: str) -> List[str]:
        """Get permissions for a role."""
        role = self._roles.get(role_name)
        return list(role.permissions) if role else []
    
    # Decorator for method-level authorization
    def require_permission(self, resource: str, action: str):
        """Decorator to require specific permission for method access."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract user_id from context (implementation depends on framework)
                user_id = kwargs.get('user_id') or getattr(args[0], 'current_user_id', None)
                if not user_id:
                    raise AuthorizationError("User ID required for authorization")
                
                result = self.check_permission(user_id, resource, action)
                if not result.allowed:
                    raise AuthorizationError(f"Access denied: {result.reason}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # Private helper methods
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user through roles."""
        user_roles = self._user_roles.get(user_id, set())
        permissions = set()
        
        for role_name in user_roles:
            role = self._roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        
        return list(permissions)
    
    def _log_access_decision(self, request: AccessRequest, result: AccessResult) -> None:
        """Log access control decision for audit."""
        if self.config.compliance.enable_audit_logging:
            self.logger.info(
                "Access control decision",
                user_id=request.user_id,
                resource=request.resource,
                action=request.action,
                allowed=result.allowed,
                reason=result.reason,
                matched_policies=result.matched_policies,
                user_roles=result.user_roles,
                timestamp=request.timestamp.isoformat()
            )
    
    def _initialize_default_permissions(self) -> None:
        """Initialize default system permissions."""
        default_perms = [
            ("user_read", "Read user information", ResourceType.USER, PermissionType.READ),
            ("user_write", "Modify user information", ResourceType.USER, PermissionType.WRITE),
            ("user_delete", "Delete users", ResourceType.USER, PermissionType.DELETE),
            ("role_read", "Read roles", ResourceType.ROLE, PermissionType.READ),
            ("role_write", "Modify roles", ResourceType.ROLE, PermissionType.WRITE),
            ("role_delete", "Delete roles", ResourceType.ROLE, PermissionType.DELETE),
            ("data_read", "Read data", ResourceType.DATA, PermissionType.READ),
            ("data_write", "Write data", ResourceType.DATA, PermissionType.WRITE),
            ("data_delete", "Delete data", ResourceType.DATA, PermissionType.DELETE),
            ("api_read", "Access read APIs", ResourceType.API, PermissionType.READ),
            ("api_write", "Access write APIs", ResourceType.API, PermissionType.WRITE),
            ("system_admin", "System administration", ResourceType.SYSTEM, PermissionType.ADMIN),
            ("audit_read", "Read audit logs", ResourceType.AUDIT, PermissionType.READ),
        ]
        
        for name, desc, resource_type, perm_type in default_perms:
            self.create_permission(name, desc, resource_type, perm_type)
    
    def _initialize_default_roles(self) -> None:
        """Initialize default system roles."""
        # Admin role with all permissions
        admin_permissions = list(self._permissions.keys())
        self.create_role(
            "admin",
            "System administrator with full access",
            admin_permissions
        )
        
        # User role with basic permissions
        user_permissions = [
            "user:read:user_read",
            "data:read:data_read",
            "api:read:api_read"
        ]
        self.create_role(
            "user",
            "Basic user with read access",
            user_permissions
        )
        
        # Editor role with read/write permissions
        editor_permissions = [
            "user:read:user_read",
            "data:read:data_read",
            "data:write:data_write",
            "api:read:api_read",
            "api:write:api_write"
        ]
        self.create_role(
            "editor",
            "Editor with read/write access",
            editor_permissions
        )