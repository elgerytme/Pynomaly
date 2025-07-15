"""GraphQL context management for Pynomaly API."""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import Request
from strawberry.types import Info

from pynomaly.domain.entities.user import User
from pynomaly.infrastructure.config import Container


class GraphQLContext:
    """GraphQL context containing request information and dependencies."""
    
    def __init__(
        self,
        request: Request,
        container: Container,
        user: Optional[User] = None,
        permissions: Optional[list[str]] = None,
        tenant_id: Optional[UUID] = None
    ):
        self.request = request
        self.container = container
        self.user = user
        self.permissions = permissions or []
        self.tenant_id = tenant_id
        self.services = {}  # Cache for services
    
    def get_service(self, service_name: str) -> Any:
        """Get a service from the container with caching."""
        if service_name not in self.services:
            self.services[service_name] = getattr(self.container, service_name)()
        return self.services[service_name]
    
    @property
    def user_id(self) -> Optional[UUID]:
        """Get current user ID."""
        return self.user.id if self.user else None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.user is not None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def require_authentication(self) -> None:
        """Require authentication, raise error if not authenticated."""
        if not self.is_authenticated:
            raise GraphQLAuthenticationError("Authentication required")
    
    def require_permission(self, permission: str) -> None:
        """Require specific permission, raise error if not available."""
        self.require_authentication()
        if not self.has_permission(permission):
            raise GraphQLPermissionError(f"Permission '{permission}' required")


class GraphQLAuthenticationError(Exception):
    """Authentication error for GraphQL."""
    pass


class GraphQLPermissionError(Exception):
    """Permission error for GraphQL."""
    pass


async def get_graphql_context(request: Request) -> GraphQLContext:
    """Create GraphQL context from FastAPI request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        GraphQL context with user information and services
    """
    # Get container from app state
    container = request.app.state.container
    
    # Initialize context
    context = GraphQLContext(request=request, container=container)
    
    # Extract user from request if available
    user = None
    permissions = []
    tenant_id = None
    
    # Check for JWT token in Authorization header
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            # Decode JWT token and extract user information
            from pynomaly.infrastructure.auth.jwt_auth import verify_jwt_token
            
            jwt_service = container.jwt_service()
            payload = await jwt_service.verify_token(token)
            
            if payload:
                user_id = payload.get("sub")
                if user_id:
                    # Get user from database
                    user_service = container.user_service()
                    user = await user_service.get_user_by_id(UUID(user_id))
                    
                    if user:
                        # Get user permissions
                        rbac_service = container.rbac_service()
                        permissions = await rbac_service.get_user_permissions(user.id)
                        tenant_id = user.tenant_id
                        
        except Exception:
            # Token verification failed, continue without user
            pass
    
    # Update context with user information
    context.user = user
    context.permissions = permissions
    context.tenant_id = tenant_id
    
    return context


def get_context_from_info(info: Info) -> GraphQLContext:
    """Extract GraphQL context from Strawberry info object.
    
    Args:
        info: Strawberry info object
        
    Returns:
        GraphQL context
    """
    return info.context