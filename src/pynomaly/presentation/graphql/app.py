"""GraphQL application factory and FastAPI integration."""

from __future__ import annotations

from typing import Dict, Any, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.extensions import QueryCache, ValidationCache
from fastapi import FastAPI, Request, Depends

from pynomaly.infrastructure.container import Container
from pynomaly.presentation.graphql.schema import schema
from pynomaly.presentation.graphql.context import (
    get_graphql_context,
    AuthenticationExtension,
    PermissionExtension,
    AuditExtension,
    RateLimitExtension,
    SecurityExtension,
)


def create_graphql_app(
    container: Container,
    websocket_manager: Optional[Any] = None,
    enable_playground: bool = True,
    enable_introspection: bool = True,
    rate_limit_requests_per_minute: int = 60,
    max_query_depth: int = 15,
    max_query_complexity: int = 1000
) -> GraphQLRouter:
    """Create and configure the GraphQL application."""
    
    # Define permission map for operations
    permission_map = {
        # User Management
        "createUser": "user.create",
        "updateUser": "user.update",
        "deleteUser": "user.delete",
        
        # Detector Management
        "createDetector": "detector.create",
        "updateDetector": "detector.update",
        "deleteDetector": "detector.delete",
        "trainDetector": "detector.train",
        
        # Dataset Management
        "createDataset": "dataset.create",
        "updateDataset": "dataset.update",
        "deleteDataset": "dataset.delete",
        
        # Detection Operations
        "runDetection": "detection.run",
        "batchDetection": "detection.batch",
        
        # Monitoring and Security
        "systemHealth": "system.monitor",
        "securityMetrics": "security.monitor",
        "auditLogs": "audit.read",
        
        # Subscriptions
        "trainingProgress": "detector.train",
        "detectionResults": "detection.read",
        "systemHealth": "system.monitor",
        "auditEvents": "audit.read",
        "performanceMetrics": "metrics.read",
        "securityMetrics": "security.monitor",
    }
    
    # Configure extensions
    extensions = [
        QueryCache(),
        ValidationCache(),
        AuthenticationExtension(),
        PermissionExtension(permission_map),
        AuditExtension(),
        RateLimitExtension(requests_per_minute=rate_limit_requests_per_minute),
        SecurityExtension(
            max_query_depth=max_query_depth,
            max_query_complexity=max_query_complexity
        ),
    ]
    
    # Create context getter with container injection
    async def context_getter(request: Request) -> Dict[str, Any]:
        return await get_graphql_context(
            request=request,
            container=container,
            websocket_manager=websocket_manager
        )
    
    # Create GraphQL router
    graphql_router = GraphQLRouter(
        schema=schema,
        context_getter=context_getter,
        extensions=extensions,
        graphiql=enable_playground,
        subscription_protocols=["graphql-ws", "graphql-transport-ws"],
        introspection=enable_introspection,
    )
    
    return graphql_router


def mount_graphql_app(
    app: FastAPI,
    container: Container,
    websocket_manager: Optional[Any] = None,
    path: str = "/graphql",
    enable_playground: bool = True,
    enable_introspection: bool = True,
    rate_limit_requests_per_minute: int = 60,
    max_query_depth: int = 15,
    max_query_complexity: int = 1000
) -> None:
    """Mount GraphQL application to FastAPI app."""
    
    graphql_router = create_graphql_app(
        container=container,
        websocket_manager=websocket_manager,
        enable_playground=enable_playground,
        enable_introspection=enable_introspection,
        rate_limit_requests_per_minute=rate_limit_requests_per_minute,
        max_query_depth=max_query_depth,
        max_query_complexity=max_query_complexity
    )
    
    # Include the GraphQL router
    app.include_router(graphql_router, prefix=path, tags=["GraphQL"])
    
    # Add GraphQL-specific middleware if needed
    @app.middleware("http")
    async def graphql_middleware(request: Request, call_next):
        """Middleware for GraphQL-specific handling."""
        
        # Add CORS headers for GraphQL
        if request.url.path.startswith(path):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response
        
        return await call_next(request)


def create_standalone_graphql_app(
    container: Container,
    websocket_manager: Optional[Any] = None,
    enable_playground: bool = True,
    enable_introspection: bool = True,
    enable_cors: bool = True,
    rate_limit_requests_per_minute: int = 60,
    max_query_depth: int = 15,
    max_query_complexity: int = 1000
) -> FastAPI:
    """Create a standalone FastAPI app with only GraphQL."""
    
    app = FastAPI(
        title="Pynomaly GraphQL API",
        description="GraphQL API for Pynomaly anomaly detection platform",
        version="1.0.0",
        docs_url="/docs" if enable_playground else None,
        redoc_url="/redoc" if enable_playground else None,
    )
    
    # Add CORS middleware if enabled
    if enable_cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Mount GraphQL
    mount_graphql_app(
        app=app,
        container=container,
        websocket_manager=websocket_manager,
        path="",  # Mount at root for standalone app
        enable_playground=enable_playground,
        enable_introspection=enable_introspection,
        rate_limit_requests_per_minute=rate_limit_requests_per_minute,
        max_query_depth=max_query_depth,
        max_query_complexity=max_query_complexity
    )
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for GraphQL app."""
        return {"status": "healthy", "service": "graphql-api"}
    
    # Add GraphQL schema endpoint
    @app.get("/schema")
    async def get_schema():
        """Get the GraphQL schema in SDL format."""
        return {"schema": str(schema)}
    
    return app


class GraphQLConfig:
    """Configuration class for GraphQL application."""
    
    def __init__(
        self,
        enable_playground: bool = True,
        enable_introspection: bool = True,
        enable_cors: bool = True,
        rate_limit_requests_per_minute: int = 60,
        max_query_depth: int = 15,
        max_query_complexity: int = 1000,
        enable_query_cache: bool = True,
        enable_validation_cache: bool = True,
        enable_authentication: bool = True,
        enable_permissions: bool = True,
        enable_audit_logging: bool = True,
        enable_rate_limiting: bool = True,
        enable_security_extensions: bool = True,
    ):
        self.enable_playground = enable_playground
        self.enable_introspection = enable_introspection
        self.enable_cors = enable_cors
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute
        self.max_query_depth = max_query_depth
        self.max_query_complexity = max_query_complexity
        self.enable_query_cache = enable_query_cache
        self.enable_validation_cache = enable_validation_cache
        self.enable_authentication = enable_authentication
        self.enable_permissions = enable_permissions
        self.enable_audit_logging = enable_audit_logging
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_security_extensions = enable_security_extensions
    
    @classmethod
    def development(cls) -> "GraphQLConfig":
        """Development configuration with all features enabled."""
        return cls(
            enable_playground=True,
            enable_introspection=True,
            enable_cors=True,
            rate_limit_requests_per_minute=1000,  # Higher limit for development
            max_query_depth=20,  # Higher depth for development
            max_query_complexity=2000,  # Higher complexity for development
        )
    
    @classmethod
    def production(cls) -> "GraphQLConfig":
        """Production configuration with security hardening."""
        return cls(
            enable_playground=False,  # Disable in production
            enable_introspection=False,  # Disable in production
            enable_cors=False,  # Configure CORS separately
            rate_limit_requests_per_minute=60,  # Strict rate limiting
            max_query_depth=10,  # Lower depth for security
            max_query_complexity=500,  # Lower complexity for security
        )
    
    @classmethod
    def testing(cls) -> "GraphQLConfig":
        """Testing configuration with relaxed limits."""
        return cls(
            enable_playground=True,
            enable_introspection=True,
            enable_cors=True,
            rate_limit_requests_per_minute=10000,  # Very high for testing
            max_query_depth=50,  # High for testing complex queries
            max_query_complexity=5000,  # High for testing
            enable_rate_limiting=False,  # Disable for testing
        )