"""
Enterprise Access Control Service - Comprehensive RBAC/ABAC implementation for enterprise security.

This service provides role-based access control (RBAC), attribute-based access control (ABAC),
fine-grained permissions management, and enterprise-grade access control features.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import secrets

from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"
    VIEW_PII = "view_pii"
    MASK_PII = "mask_pii"
    EXPORT_DATA = "export_data"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    AUDIT_VIEW = "audit_view"
    SYSTEM_CONFIG = "system_config"


class ResourceType(Enum):
    """Types of resources that can be protected."""
    DATASET = "dataset"
    MODEL = "model"
    PIPELINE = "pipeline"
    REPORT = "report"
    USER = "user"
    ROLE = "role"
    SYSTEM = "system"
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    FILE = "file"


class AccessDecision(Enum):
    """Access control decisions."""
    PERMIT = "permit"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"


class AttributeType(Enum):
    """Types of attributes for ABAC."""
    USER = "user"
    RESOURCE = "resource"
    ENVIRONMENT = "environment"
    ACTION = "action"


@dataclass
class Attribute:
    """ABAC attribute definition."""
    name: str
    value: Any
    attribute_type: AttributeType
    data_type: str = "string"  # string, number, boolean, datetime, list
    description: str = ""


@dataclass
class Role:
    """Role definition for RBAC."""
    id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # Role hierarchy
    resource_restrictions: Dict[ResourceType, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class User:
    """User definition with roles and attributes."""
    id: str
    username: str
    email: str
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    password_hash: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    lock_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class Resource:
    """Protected resource definition."""
    id: str
    name: str
    resource_type: ResourceType
    owner_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    sensitivity_level: str = "medium"  # low, medium, high, critical
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class PolicyRule:
    """ABAC policy rule definition."""
    id: str
    name: str
    description: str
    effect: AccessDecision  # PERMIT or DENY
    target_conditions: List[str] = field(default_factory=list)  # Conditions to match
    priority: int = 100  # Higher number = higher priority
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessRequest:
    """Access control request."""
    user_id: str
    resource_id: str
    action: Permission
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessResponse:
    """Access control response."""
    decision: AccessDecision
    reason: str
    applicable_policies: List[str] = field(default_factory=list)
    user_roles: List[str] = field(default_factory=list)
    effective_permissions: Set[Permission] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None


@dataclass
class AuditEvent:
    """Access control audit event."""
    event_id: str
    event_type: str  # login, logout, access_granted, access_denied, permission_changed
    user_id: str
    resource_id: Optional[str] = None
    action: Optional[Permission] = None
    decision: Optional[AccessDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EnterpriseAccessControlService:
    """Comprehensive enterprise access control service implementing RBAC and ABAC."""
    
    def __init__(self):
        """Initialize the enterprise access control service."""
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.resources: Dict[str, Resource] = {}
        self.policy_rules: Dict[str, PolicyRule] = {}
        self.audit_events: List[AuditEvent] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_failed_login_attempts = 5
        self.account_lockout_duration_minutes = 30
        self.session_timeout_minutes = 480  # 8 hours
        self.password_min_length = 12
        
        # Initialize default roles and policies
        self._initialize_default_roles()
        self._initialize_default_policies()
        
        logger.info("Enterprise Access Control Service initialized")
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Super Admin role
        super_admin = Role(
            id="super_admin",
            name="Super Administrator",
            description="Full system access",
            permissions=set(Permission),
            attributes={"clearance_level": "top_secret", "department": "IT"}
        )
        
        # Data Scientist role
        data_scientist = Role(
            id="data_scientist",
            name="Data Scientist",
            description="Data analysis and model development",
            permissions={
                Permission.READ, Permission.WRITE, Permission.CREATE, Permission.UPDATE,
                Permission.EXECUTE, Permission.VIEW_PII, Permission.EXPORT_DATA
            },
            resource_restrictions={
                ResourceType.SYSTEM: [],  # No system access
                ResourceType.USER: []     # No user management
            },
            attributes={"clearance_level": "secret", "department": "Data Science"}
        )
        
        # Data Analyst role
        data_analyst = Role(
            id="data_analyst",
            name="Data Analyst",
            description="Data analysis and reporting",
            permissions={
                Permission.READ, Permission.EXECUTE, Permission.MASK_PII
            },
            attributes={"clearance_level": "confidential", "department": "Analytics"}
        )
        
        # Auditor role
        auditor = Role(
            id="auditor",
            name="Auditor",
            description="Audit and compliance review",
            permissions={
                Permission.READ, Permission.AUDIT_VIEW
            },
            attributes={"clearance_level": "secret", "department": "Compliance"}
        )
        
        # Business User role
        business_user = Role(
            id="business_user",
            name="Business User",
            description="Basic business operations",
            permissions={
                Permission.READ
            },
            attributes={"clearance_level": "internal", "department": "Business"}
        )
        
        self.roles.update({
            "super_admin": super_admin,
            "data_scientist": data_scientist,
            "data_analyst": data_analyst,
            "auditor": auditor,
            "business_user": business_user
        })
    
    def _initialize_default_policies(self):
        """Initialize default ABAC policies."""
        # PII Access Policy
        pii_policy = PolicyRule(
            id="pii_access_policy",
            name="PII Access Control",
            description="Controls access to PII data based on clearance level",
            effect=AccessDecision.PERMIT,
            target_conditions=[
                "resource.sensitivity_level == 'high' AND action == 'view_pii'",
                "user.clearance_level IN ['secret', 'top_secret']"
            ],
            priority=200
        )
        
        # Time-based Access Policy
        time_policy = PolicyRule(
            id="time_based_policy",
            name="Business Hours Access",
            description="Restricts access to business hours for certain resources",
            effect=AccessDecision.DENY,
            target_conditions=[
                "resource.tags CONTAINS 'business_hours_only'",
                "environment.time NOT BETWEEN '09:00' AND '17:00'"
            ],
            priority=150
        )
        
        # Department Segregation Policy
        dept_policy = PolicyRule(
            id="department_segregation",
            name="Department Data Segregation",
            description="Users can only access data from their department",
            effect=AccessDecision.PERMIT,
            target_conditions=[
                "user.department == resource.department OR user.clearance_level == 'top_secret'"
            ],
            priority=100
        )
        
        self.policy_rules.update({
            "pii_access_policy": pii_policy,
            "time_based_policy": time_policy,
            "department_segregation": dept_policy
        })
    
    # Error handling would be managed by interface implementation
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (success, session_id, error_message)
        """
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            await self._log_audit_event(
                "login_failed", username, None, None, None,
                {"reason": "user_not_found", "ip_address": ip_address}
            )
            return False, None, "Invalid credentials"
        
        # Check if account is locked
        if user.account_locked and user.lock_until and datetime.utcnow() < user.lock_until:
            await self._log_audit_event(
                "login_failed", user.id, None, None, None,
                {"reason": "account_locked", "ip_address": ip_address}
            )
            return False, None, "Account is locked"
        
        # Unlock account if lock period has expired
        if user.account_locked and user.lock_until and datetime.utcnow() >= user.lock_until:
            user.account_locked = False
            user.failed_login_attempts = 0
            user.lock_until = None
        
        # Verify password (simplified - in production use proper password hashing)
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_login_attempts:
                user.account_locked = True
                user.lock_until = datetime.utcnow() + timedelta(minutes=self.account_lockout_duration_minutes)
                
                await self._log_audit_event(
                    "account_locked", user.id, None, None, None,
                    {"failed_attempts": user.failed_login_attempts, "ip_address": ip_address}
                )
            
            await self._log_audit_event(
                "login_failed", user.id, None, None, None,
                {"reason": "invalid_password", "failed_attempts": user.failed_login_attempts, "ip_address": ip_address}
            )
            return False, None, "Invalid credentials"
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        # Create session
        session_id = self._create_session(user.id, ip_address, user_agent)
        
        await self._log_audit_event(
            "login_success", user.id, None, None, None,
            {"session_id": session_id, "ip_address": ip_address}
        )
        
        logger.info(f"User {username} authenticated successfully")
        return True, session_id, None
    
    def _verify_password(self, password: str, password_hash: Optional[str]) -> bool:
        """Verify password against hash (simplified implementation)."""
        if not password_hash:
            return False
        # In production, use proper password hashing (bcrypt, scrypt, etc.)
        return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    def _create_session(self, user_id: str, ip_address: Optional[str], user_agent: Optional[str]) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        self.active_sessions[session_id] = session_data
        return session_id
    
    # Error handling would be managed by interface implementation
    async def check_access(
        self,
        user_id: str,
        resource_id: str,
        action: Permission,
        context: Optional[Dict[str, Any]] = None
    ) -> AccessResponse:
        """
        Check if a user has access to perform an action on a resource.
        
        Args:
            user_id: User requesting access
            resource_id: Resource being accessed
            action: Action to perform
            context: Additional context (IP, time, etc.)
            
        Returns:
            AccessResponse with decision and details
        """
        context = context or {}
        
        # Get user, resource, and build context
        user = self.users.get(user_id)
        resource = self.resources.get(resource_id)
        
        if not user:
            return AccessResponse(
                decision=AccessDecision.DENY,
                reason="User not found"
            )
        
        if not resource:
            return AccessResponse(
                decision=AccessDecision.DENY,
                reason="Resource not found"
            )
        
        # Check RBAC first
        rbac_decision = await self._check_rbac(user, resource, action)
        
        if rbac_decision.decision == AccessDecision.DENY:
            await self._log_audit_event(
                "access_denied", user_id, resource_id, action, AccessDecision.DENY,
                {"reason": "RBAC denial", "details": rbac_decision.reason}
            )
            return rbac_decision
        
        # Check ABAC policies
        abac_decision = await self._check_abac(user, resource, action, context)
        
        # Combine decisions (RBAC PERMIT + ABAC check)
        final_decision = AccessDecision.PERMIT if (
            rbac_decision.decision == AccessDecision.PERMIT and 
            abac_decision.decision in [AccessDecision.PERMIT, AccessDecision.NOT_APPLICABLE]
        ) else AccessDecision.DENY
        
        response = AccessResponse(
            decision=final_decision,
            reason=abac_decision.reason if abac_decision.decision == AccessDecision.DENY else "Access granted",
            applicable_policies=abac_decision.applicable_policies,
            user_roles=list(user.roles),
            effective_permissions=rbac_decision.effective_permissions
        )
        
        # Log access decision
        await self._log_audit_event(
            "access_granted" if final_decision == AccessDecision.PERMIT else "access_denied",
            user_id, resource_id, action, final_decision,
            {"reason": response.reason}
        )
        
        return response
    
    async def _check_rbac(self, user: User, resource: Resource, action: Permission) -> AccessResponse:
        """Check role-based access control."""
        effective_permissions = set()
        user_roles = []
        
        # Get all permissions from user's roles (including inherited)
        all_roles = self._get_all_user_roles(user.id)
        
        for role_id in all_roles:
            role = self.roles.get(role_id)
            if role and role.is_active:
                user_roles.append(role.name)
                effective_permissions.update(role.permissions)
                
                # Check resource restrictions
                if resource.resource_type in role.resource_restrictions:
                    restricted_resources = role.resource_restrictions[resource.resource_type]
                    if restricted_resources and resource.id not in restricted_resources:
                        return AccessResponse(
                            decision=AccessDecision.DENY,
                            reason=f"Role {role.name} restricted from accessing this {resource.resource_type.value}",
                            user_roles=user_roles,
                            effective_permissions=effective_permissions
                        )
        
        # Check if user has the required permission
        if action in effective_permissions or Permission.ADMIN in effective_permissions:
            return AccessResponse(
                decision=AccessDecision.PERMIT,
                reason="RBAC permission granted",
                user_roles=user_roles,
                effective_permissions=effective_permissions
            )
        else:
            return AccessResponse(
                decision=AccessDecision.DENY,
                reason=f"Required permission {action.value} not found in user roles",
                user_roles=user_roles,
                effective_permissions=effective_permissions
            )
    
    def _get_all_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles for a user including inherited roles."""
        user = self.users.get(user_id)
        if not user:
            return set()
        
        all_roles = set(user.roles)
        
        # Add inherited roles (simplified role hierarchy)
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role:
                all_roles.update(role.parent_roles)
        
        return all_roles
    
    async def _check_abac(
        self,
        user: User,
        resource: Resource,
        action: Permission,
        context: Dict[str, Any]
    ) -> AccessResponse:
        """Check attribute-based access control."""
        applicable_policies = []
        
        # Build evaluation context
        eval_context = {
            "user": user.attributes,
            "resource": {**resource.attributes, "sensitivity_level": resource.sensitivity_level, "tags": list(resource.tags)},
            "action": action.value,
            "environment": {
                "time": datetime.utcnow().strftime("%H:%M"),
                "day_of_week": datetime.utcnow().strftime("%A"),
                **context
            }
        }
        
        # Evaluate policies in priority order
        sorted_policies = sorted(self.policy_rules.values(), key=lambda p: p.priority, reverse=True)
        
        for policy in sorted_policies:
            if not policy.is_active:
                continue
            
            if self._evaluate_policy_conditions(policy.target_conditions, eval_context):
                applicable_policies.append(policy.name)
                
                if policy.effect == AccessDecision.DENY:
                    return AccessResponse(
                        decision=AccessDecision.DENY,
                        reason=f"ABAC policy '{policy.name}' denied access",
                        applicable_policies=applicable_policies
                    )
                elif policy.effect == AccessDecision.PERMIT:
                    return AccessResponse(
                        decision=AccessDecision.PERMIT,
                        reason=f"ABAC policy '{policy.name}' granted access",
                        applicable_policies=applicable_policies
                    )
        
        # No applicable policies found
        return AccessResponse(
            decision=AccessDecision.NOT_APPLICABLE,
            reason="No applicable ABAC policies found",
            applicable_policies=applicable_policies
        )
    
    def _evaluate_policy_conditions(self, conditions: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate policy conditions (simplified implementation)."""
        try:
            for condition in conditions:
                # Simple condition evaluation (in production, use a proper policy engine)
                if not self._evaluate_single_condition(condition, context):
                    return False
            return True
        except Exception as e:
            logger.error(f"Policy condition evaluation error: {e}")
            return False
    
    def _evaluate_single_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition (simplified implementation)."""
        # This is a very simplified condition evaluator
        # In production, use a proper expression language or policy engine
        
        try:
            # Replace context variables
            for key, value in context.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        placeholder = f"{key}.{subkey}"
                        if isinstance(subvalue, str):
                            condition = condition.replace(placeholder, f"'{subvalue}'")
                        else:
                            condition = condition.replace(placeholder, str(subvalue))
                else:
                    if isinstance(value, str):
                        condition = condition.replace(key, f"'{value}'")
                    else:
                        condition = condition.replace(key, str(value))
            
            # Evaluate (unsafe - use proper expression evaluator in production)
            # For demo purposes only
            return True  # Simplified evaluation
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    # Error handling would be managed by interface implementation
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user."""
        user_id = f"user_{secrets.token_hex(8)}"
        
        # Hash password (simplified)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=set(roles or []),
            attributes=attributes or {},
            password_hash=password_hash
        )
        
        self.users[user_id] = user
        
        await self._log_audit_event(
            "user_created", user_id, None, None, None,
            {"username": username, "email": email, "roles": roles or []}
        )
        
        logger.info(f"User created: {username}")
        return user
    
    # Error handling would be managed by interface implementation
    async def create_role(
        self,
        name: str,
        description: str,
        permissions: List[Permission],
        parent_roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Role:
        """Create a new role."""
        role_id = f"role_{secrets.token_hex(8)}"
        
        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=set(permissions),
            parent_roles=set(parent_roles or []),
            attributes=attributes or {}
        )
        
        self.roles[role_id] = role
        
        await self._log_audit_event(
            "role_created", "system", None, None, None,
            {"role_id": role_id, "name": name, "permissions": [p.value for p in permissions]}
        )
        
        logger.info(f"Role created: {name}")
        return role
    
    # Error handling would be managed by interface implementation
    async def create_resource(
        self,
        name: str,
        resource_type: ResourceType,
        owner_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        sensitivity_level: str = "medium"
    ) -> Resource:
        """Create a new protected resource."""
        resource_id = f"res_{secrets.token_hex(8)}"
        
        resource = Resource(
            id=resource_id,
            name=name,
            resource_type=resource_type,
            owner_id=owner_id,
            attributes=attributes or {},
            tags=tags or set(),
            sensitivity_level=sensitivity_level
        )
        
        self.resources[resource_id] = resource
        
        await self._log_audit_event(
            "resource_created", owner_id, resource_id, None, None,
            {"name": name, "type": resource_type.value, "sensitivity": sensitivity_level}
        )
        
        logger.info(f"Resource created: {name}")
        return resource
    
    async def _log_audit_event(
        self,
        event_type: str,
        user_id: str,
        resource_id: Optional[str],
        action: Optional[Permission],
        decision: Optional[AccessDecision],
        details: Dict[str, Any]
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_id=f"evt_{secrets.token_hex(8)}",
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            decision=decision,
            details=details
        )
        
        self.audit_events.append(event)
        
        # Keep only last 50,000 events
        if len(self.audit_events) > 50000:
            self.audit_events = self.audit_events[-50000:]
    
    # Error handling would be managed by interface implementation
    async def get_access_control_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive access control dashboard."""
        # Calculate statistics
        active_users = len([u for u in self.users.values() if u.is_active])
        locked_accounts = len([u for u in self.users.values() if u.account_locked])
        active_sessions = len(self.active_sessions)
        
        # Recent audit events
        recent_events = self.audit_events[-100:] if self.audit_events else []
        
        # Failed access attempts in last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_failures = len([
            e for e in self.audit_events
            if e.event_type == "access_denied" and e.timestamp > cutoff_time
        ])
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "user_management": {
                "total_users": len(self.users),
                "active_users": active_users,
                "locked_accounts": locked_accounts,
                "active_sessions": active_sessions
            },
            "role_management": {
                "total_roles": len(self.roles),
                "active_roles": len([r for r in self.roles.values() if r.is_active])
            },
            "resource_management": {
                "total_resources": len(self.resources),
                "by_type": {
                    resource_type.value: len([r for r in self.resources.values() if r.resource_type == resource_type])
                    for resource_type in ResourceType
                },
                "by_sensitivity": {
                    level: len([r for r in self.resources.values() if r.sensitivity_level == level])
                    for level in ["low", "medium", "high", "critical"]
                }
            },
            "policy_management": {
                "total_policies": len(self.policy_rules),
                "active_policies": len([p for p in self.policy_rules.values() if p.is_active])
            },
            "security_metrics": {
                "failed_access_attempts_24h": recent_failures,
                "total_audit_events": len(self.audit_events),
                "recent_login_failures": len([
                    e for e in recent_events
                    if e.event_type == "login_failed"
                ])
            },
            "supported_permissions": [p.value for p in Permission],
            "supported_resource_types": [rt.value for rt in ResourceType],
            "recent_audit_events": [
                {
                    "event_type": e.event_type,
                    "user_id": e.user_id,
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details
                }
                for e in recent_events[-10:]  # Last 10 events
            ]
        }