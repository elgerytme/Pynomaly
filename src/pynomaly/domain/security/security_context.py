"""Security context domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field


class SecurityContext(BaseModel):
    """Security context for operations."""
    
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    permissions: Set[str] = Field(default_factory=set)
    roles: List[str] = Field(default_factory=list)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authenticated: bool = False
    authentication_method: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if context has specific role."""
        return role in self.roles
    
    def is_expired(self) -> bool:
        """Check if security context is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if security context is valid."""
        return self.authenticated and not self.is_expired()