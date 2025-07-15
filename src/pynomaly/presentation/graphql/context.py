"""GraphQL context for dependency injection and authentication."""

from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pynomaly.application.services.auth_service import AuthenticationService
from pynomaly.domain.entities.user import User
from pynomaly.infrastructure.container import Container

if TYPE_CHECKING:
    from strawberry.fastapi import GraphQLRouter


class GraphQLContext:
    """GraphQL context for dependency injection and user authentication."""
    
    def __init__(
        self,
        request: Request,
        container: Container,
        user: Optional[User] = None,
        websocket_manager: Optional[Any] = None
    ):
        self.request = request
        self.container = container
        self.user = user
        self.websocket_manager = websocket_manager
        self._services_cache: Dict[str, Any] = {}
    
    def get_service(self, service_class: type):
        """Get a service from the container with caching."""
        service_name = service_class.__name__
        
        if service_name not in self._services_cache:
            self._services_cache[service_name] = self.container.get(service_class)
        
        return self._services_cache[service_name]
    
    def get_user(self) -> Optional[User]:
        """Get the authenticated user."""
        return self.user
    
    def require_user(self) -> User:
        """Get the authenticated user or raise an error."""
        if not self.user:
            raise HTTPException(status_code=401, detail="Authentication required")
        return self.user
    
    def require_permission(self, permission: str) -> User:
        """Require user to have specific permission."""
        user = self.require_user()
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403, 
                detail=f"Permission '{permission}' required"
            )
        return user
    
    def get_request_metadata(self) -> Dict[str, Any]:
        """Get request metadata for audit logging."""
        return {
            "ip_address": self.request.client.host if self.request.client else None,
            "user_agent": self.request.headers.get("user-agent"),
            "method": self.request.method,
            "url": str(self.request.url),
            "headers": dict(self.request.headers)
        }


async def get_graphql_context(
    request: Request,
    container: Container,
    websocket_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """Create GraphQL context with authentication and dependency injection."""
    
    # Extract JWT token from request
    user = None
    authorization = request.headers.get("Authorization")
    
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        
        try:
            auth_service = container.get(AuthenticationService)
            user = await auth_service.verify_token(token)
        except Exception:
            # Token verification failed, user remains None
            pass
    
    # Create context
    context = GraphQLContext(
        request=request,
        container=container,
        user=user,
        websocket_manager=websocket_manager
    )
    
    # Return context as dictionary for strawberry compatibility
    return {
        "request": request,
        "container": container,
        "user": user,
        "websocket_manager": websocket_manager,
        "context": context,
        # Direct access to common services
        "auth_service": container.get(AuthenticationService),
        # Add more services as needed
    }


def require_authentication(info) -> User:
    """Strawberry directive for requiring authentication."""
    user = info.context.get("user")
    if not user:
        raise ValueError("Authentication required")
    return user


def require_permission(permission: str):
    """Strawberry directive for requiring specific permission."""
    def decorator(info):
        user = require_authentication(info)
        if not user.has_permission(permission):
            raise ValueError(f"Permission '{permission}' required")
        return user
    return decorator


class AuthenticationExtension:
    """Strawberry extension for handling authentication."""
    
    def on_operation(self, context, info):
        """Handle authentication for each GraphQL operation."""
        # Check if operation requires authentication
        operation_name = info.operation.name.value if info.operation.name else "anonymous"
        
        # Skip authentication for introspection queries
        if operation_name in ["IntrospectionQuery", "__schema", "__type"]:
            return
        
        # For mutations and some queries, require authentication
        if info.operation.operation == "mutation":
            require_authentication(info)
        
        # For subscriptions, always require authentication
        if info.operation.operation == "subscription":
            require_authentication(info)


class PermissionExtension:
    """Strawberry extension for handling permissions."""
    
    def __init__(self, permission_map: Dict[str, str]):
        """Initialize with a map of operation names to required permissions."""
        self.permission_map = permission_map
    
    def on_operation(self, context, info):
        """Check permissions for each GraphQL operation."""
        operation_name = info.operation.name.value if info.operation.name else None
        
        if operation_name in self.permission_map:
            required_permission = self.permission_map[operation_name]
            user = require_authentication(info)
            
            if not user.has_permission(required_permission):
                raise ValueError(f"Permission '{required_permission}' required for operation '{operation_name}'")


class AuditExtension:
    """Strawberry extension for audit logging."""
    
    def on_operation(self, context, info):
        """Log GraphQL operations for audit purposes."""
        user = context.get("user")
        request = context.get("request")
        
        # Create audit log entry
        audit_data = {
            "operation_type": str(info.operation.operation),
            "operation_name": info.operation.name.value if info.operation.name else "anonymous",
            "user_id": str(user.id) if user else None,
            "tenant_id": str(user.tenant_id) if user else None,
            "ip_address": request.client.host if request and request.client else None,
            "user_agent": request.headers.get("user-agent") if request else None,
            "query": info.query,
            "variables": info.variable_values
        }
        
        # Log to audit service (implement as needed)
        # audit_service = context.get("audit_service")
        # if audit_service:
        #     await audit_service.log_graphql_operation(audit_data)


class RateLimitExtension:
    """Strawberry extension for rate limiting."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, int] = {}
    
    def on_operation(self, context, info):
        """Apply rate limiting to GraphQL operations."""
        user = context.get("user")
        request = context.get("request")
        
        # Create rate limit key
        if user:
            key = f"user:{user.id}"
        elif request and request.client:
            key = f"ip:{request.client.host}"
        else:
            key = "anonymous"
        
        # Check rate limit (simplified implementation)
        current_count = self.request_counts.get(key, 0)
        if current_count >= self.requests_per_minute:
            raise ValueError("Rate limit exceeded")
        
        # Increment count
        self.request_counts[key] = current_count + 1


class SecurityExtension:
    """Strawberry extension for additional security measures."""
    
    def __init__(self, max_query_depth: int = 15, max_query_complexity: int = 1000):
        self.max_query_depth = max_query_depth
        self.max_query_complexity = max_query_complexity
    
    def on_operation(self, context, info):
        """Apply security measures to GraphQL operations."""
        # Query depth analysis (simplified)
        query_depth = self._calculate_query_depth(info.field_nodes)
        if query_depth > self.max_query_depth:
            raise ValueError(f"Query depth {query_depth} exceeds maximum {self.max_query_depth}")
        
        # Query complexity analysis (simplified)
        query_complexity = self._calculate_query_complexity(info.field_nodes)
        if query_complexity > self.max_query_complexity:
            raise ValueError(f"Query complexity {query_complexity} exceeds maximum {self.max_query_complexity}")
    
    def _calculate_query_depth(self, field_nodes, current_depth=0):
        """Calculate the depth of a GraphQL query."""
        max_depth = current_depth
        
        for field_node in field_nodes:
            if hasattr(field_node, 'selection_set') and field_node.selection_set:
                depth = self._calculate_query_depth(
                    field_node.selection_set.selections,
                    current_depth + 1
                )
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_query_complexity(self, field_nodes, current_complexity=0):
        """Calculate the complexity of a GraphQL query."""
        complexity = current_complexity
        
        for field_node in field_nodes:
            # Each field adds 1 to complexity
            complexity += 1
            
            # Nested fields add additional complexity
            if hasattr(field_node, 'selection_set') and field_node.selection_set:
                complexity += self._calculate_query_complexity(
                    field_node.selection_set.selections,
                    0
                )
        
        return complexity