"""GraphQL middleware for Pynomaly API."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from strawberry.types import Info

from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.graphql.context import (
    GraphQLAuthenticationError,
    GraphQLContext,
    GraphQLPermissionError,
)


class GraphQLErrorMiddleware(BaseHTTPMiddleware):
    """Middleware for handling GraphQL errors."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle GraphQL errors and format responses."""
        try:
            response = await call_next(request)
            return response
        except GraphQLAuthenticationError as e:
            # Handle authentication errors
            return Response(
                content=f'{{"errors": [{{"message": "{str(e)}", "extensions": {{"code": "UNAUTHENTICATED"}}}}]}}',
                status_code=401,
                media_type="application/json"
            )
        except GraphQLPermissionError as e:
            # Handle permission errors
            return Response(
                content=f'{{"errors": [{{"message": "{str(e)}", "extensions": {{"code": "FORBIDDEN"}}}}]}}',
                status_code=403,
                media_type="application/json"
            )
        except Exception as e:
            # Handle general errors
            return Response(
                content=f'{{"errors": [{{"message": "Internal server error", "extensions": {{"code": "INTERNAL_ERROR"}}}}]}}',
                status_code=500,
                media_type="application/json"
            )


class GraphQLAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for GraphQL authentication."""
    
    def __init__(self, app, container: Container):
        super().__init__(app)
        self.container = container
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add authentication information to request."""
        # Authentication is handled in context creation
        # This middleware can be used for additional auth logic
        response = await call_next(request)
        return response


class GraphQLRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for GraphQL endpoints."""
    
    def __init__(self, app, settings):
        super().__init__(app)
        self.settings = settings
        self.rate_limit_store = {}  # Simple in-memory store
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to GraphQL requests."""
        # Check if this is a GraphQL request
        if not request.url.path.startswith("/api/v1/graphql"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        current_time = time.time()
        
        # Simple sliding window rate limiting
        if client_ip in self.rate_limit_store:
            requests = self.rate_limit_store[client_ip]
            # Remove old requests (older than 1 minute)
            requests = [req_time for req_time in requests if current_time - req_time < 60]
            
            # Check rate limit (e.g., 100 requests per minute)
            if len(requests) >= 100:
                return Response(
                    content='{"errors": [{"message": "Rate limit exceeded", "extensions": {"code": "RATE_LIMITED"}}]}',
                    status_code=429,
                    media_type="application/json"
                )
            
            requests.append(current_time)
            self.rate_limit_store[client_ip] = requests
        else:
            self.rate_limit_store[client_ip] = [current_time]
        
        response = await call_next(request)
        return response


class GraphQLMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting GraphQL metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_operation": {},
            "response_times": [],
            "errors_total": 0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for GraphQL requests."""
        if not request.url.path.startswith("/api/v1/graphql"):
            return await call_next(request)
        
        start_time = time.time()
        
        # Get operation name from request body if available
        operation_name = "unknown"
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    import json
                    data = json.loads(body)
                    operation_name = data.get("operationName", "unknown")
            except Exception:
                pass
        
        try:
            response = await call_next(request)
            
            # Update metrics
            self.metrics["requests_total"] += 1
            self.metrics["requests_by_operation"][operation_name] = (
                self.metrics["requests_by_operation"].get(operation_name, 0) + 1
            )
            
            # Record response time
            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)
            
            # Keep only last 1000 response times
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
            # Check for errors in response
            if response.status_code >= 400:
                self.metrics["errors_total"] += 1
            
            return response
            
        except Exception as e:
            self.metrics["errors_total"] += 1
            raise


def require_authentication(resolver: Callable) -> Callable:
    """Decorator to require authentication for GraphQL resolvers."""
    
    async def wrapper(self, info: Info, *args, **kwargs):
        from pynomaly.presentation.api.graphql.context import get_context_from_info
        
        context = get_context_from_info(info)
        context.require_authentication()
        
        return await resolver(self, info, *args, **kwargs)
    
    return wrapper


def require_permission(permission: str) -> Callable:
    """Decorator to require specific permission for GraphQL resolvers."""
    
    def decorator(resolver: Callable) -> Callable:
        async def wrapper(self, info: Info, *args, **kwargs):
            from pynomaly.presentation.api.graphql.context import get_context_from_info
            
            context = get_context_from_info(info)
            context.require_permission(permission)
            
            return await resolver(self, info, *args, **kwargs)
        
        return wrapper
    
    return decorator


def require_role(role: str) -> Callable:
    """Decorator to require specific role for GraphQL resolvers."""
    
    def decorator(resolver: Callable) -> Callable:
        async def wrapper(self, info: Info, *args, **kwargs):
            from pynomaly.presentation.api.graphql.context import get_context_from_info
            
            context = get_context_from_info(info)
            context.require_authentication()
            
            if context.user.role != role:
                raise GraphQLPermissionError(f"Role '{role}' required")
            
            return await resolver(self, info, *args, **kwargs)
        
        return wrapper
    
    return decorator


def log_graphql_query(resolver: Callable) -> Callable:
    """Decorator to log GraphQL queries for debugging."""
    
    async def wrapper(self, info: Info, *args, **kwargs):
        # Log query information
        operation_name = info.operation.name.value if info.operation.name else "unknown"
        field_name = info.field_name
        
        print(f"GraphQL: {operation_name}.{field_name} called")
        
        try:
            result = await resolver(self, info, *args, **kwargs)
            print(f"GraphQL: {operation_name}.{field_name} completed successfully")
            return result
        except Exception as e:
            print(f"GraphQL: {operation_name}.{field_name} failed with error: {e}")
            raise
    
    return wrapper


def cache_resolver(ttl_seconds: int = 60) -> Callable:
    """Decorator to cache GraphQL resolver results."""
    
    def decorator(resolver: Callable) -> Callable:
        cache = {}
        
        async def wrapper(self, info: Info, *args, **kwargs):
            # Create cache key from resolver name and arguments
            cache_key = f"{resolver.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()
            
            # Check if result is cached and not expired
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Call resolver and cache result
            result = await resolver(self, info, *args, **kwargs)
            cache[cache_key] = (result, current_time)
            
            # Clean up old cache entries
            if len(cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:100]:
                    del cache[key]
            
            return result
        
        return wrapper
    
    return decorator