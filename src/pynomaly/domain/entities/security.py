"""Security and compliance domain entities."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Set
from uuid import UUID, uuid4


class ActionType(Enum):
    """Types of actions that can be audited."""
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    CREATE_MODEL = "create_model"
    UPDATE_MODEL = "update_model"
    DELETE_MODEL = "delete_model"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    DEPLOY_MODEL = "deploy_model"
    ACCESS_DENIED = "access_denied"


class PermissionType(Enum):
    """Types of permissions in the system."""
    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    EXECUTE_DEPLOYMENT = "execute_deployment"
    ADMIN_SYSTEM = "admin_system"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    MANAGE_USERS = "manage_users"


@dataclass
class AuditEvent:
    """Audit event entity for tracking system actions."""
    
    event_id: UUID
    user_id: UUID
    username: str
    action: ActionType
    resource_type: str
    resource_id: Optional[str]
    resource_name: Optional[str]
    ip_address: str
    success: bool
    
    # Optional fields
    timestamp: datetime = field(default_factory=datetime.utcnow)
    security_level: str = "MEDIUM"
    requires_retention: bool = True
    additional_data: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate audit event data."""
        valid_security_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if self.security_level not in valid_security_levels:
            raise ValueError(f"Security level must be one of {valid_security_levels}")
    
    def get_retention_period_days(self) -> int:
        """Get retention period in days based on security level."""
        if not self.requires_retention:
            return 30
        
        retention_map = {
            "LOW": 365,
            "MEDIUM": 1095,  # 3 years
            "HIGH": 2555,    # 7 years
            "CRITICAL": 2555  # 7 years
        }
        return retention_map.get(self.security_level, 1095)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    policy_id: UUID = field(default_factory=uuid4)
    policy_name: str = "Default Security Policy"
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    max_failed_login_attempts: int = 5
    session_timeout_minutes: int = 30
    require_2fa: bool = False
    
    def __post_init__(self):
        """Validate security policy."""
        if self.password_min_length < 8:
            raise ValueError("Password minimum length must be at least 8")
        if self.max_failed_login_attempts <= 0:
            raise ValueError("Max failed login attempts must be positive")


@dataclass
class AccessRequest:
    """Access request entity for permission management."""
    
    request_id: UUID
    requester_id: UUID
    requester_username: str
    requested_permission: PermissionType
    resource_type: str
    justification: str
    
    # Optional fields
    approval_status: str = "pending"
    approver_id: Optional[UUID] = None
    approver_username: Optional[str] = None
    approved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate access request."""
        if not self.justification.strip():
            raise ValueError("Justification cannot be empty")


@dataclass
class User:
    """User entity for authentication and authorization."""
    
    user_id: UUID
    username: str
    email: str
    password_hash: str
    salt: str
    
    # Optional fields
    roles: Set[str] = field(default_factory=set)
    custom_permissions: Set[PermissionType] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def has_permission(self, permission: PermissionType) -> bool:
        """Check if user has a specific permission."""
        # Admin role has all permissions
        if "admin" in self.roles:
            return True
        
        # Check custom permissions
        if permission in self.custom_permissions:
            return True
        
        # Check role-based permissions
        role_permissions = {
            "user": {PermissionType.READ_MODELS},
            "analyst": {PermissionType.READ_MODELS, PermissionType.WRITE_MODELS},
            "deployer": {PermissionType.READ_MODELS, PermissionType.EXECUTE_DEPLOYMENT},
            "admin": set(PermissionType),  # All permissions
        }
        
        for role in self.roles:
            if role in role_permissions and permission in role_permissions[role]:
                return True
        
        return False
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
