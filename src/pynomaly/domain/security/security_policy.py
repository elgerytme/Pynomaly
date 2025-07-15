"""Security policy domain model."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

from pydantic import BaseModel, Field


class SecurityLevel(str, Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessControl(BaseModel):
    """Access control configuration."""
    
    allowed_operations: Set[str] = Field(default_factory=set)
    denied_operations: Set[str] = Field(default_factory=set)
    required_permissions: Set[str] = Field(default_factory=set)
    required_roles: Set[str] = Field(default_factory=set)
    ip_whitelist: List[str] = Field(default_factory=list)
    ip_blacklist: List[str] = Field(default_factory=list)


class SecurityPolicy(BaseModel):
    """Security policy configuration."""
    
    name: str
    description: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    access_control: AccessControl = Field(default_factory=AccessControl)
    session_timeout: timedelta = Field(default=timedelta(hours=8))
    max_failed_attempts: int = 5
    lockout_duration: timedelta = Field(default=timedelta(minutes=30))
    require_mfa: bool = False
    password_policy: Dict[str, bool] = Field(default_factory=lambda: {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True
    })
    audit_all_actions: bool = True
    log_failed_attempts: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)
    
    def allows_operation(self, operation: str) -> bool:
        """Check if policy allows specific operation."""
        if operation in self.access_control.denied_operations:
            return False
        if self.access_control.allowed_operations:
            return operation in self.access_control.allowed_operations
        return True
    
    def get_required_permissions(self) -> Set[str]:
        """Get required permissions for this policy."""
        return self.access_control.required_permissions
    
    def get_required_roles(self) -> Set[str]:
        """Get required roles for this policy."""
        return self.access_control.required_roles