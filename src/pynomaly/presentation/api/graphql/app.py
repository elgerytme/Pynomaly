"""GraphQL application setup for Pynomaly."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from graphql import ExecutionResult
from strawberry.fastapi import GraphQLRouter

from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.graphql.context import get_graphql_context
from pynomaly.presentation.api.graphql.middleware import (
    GraphQLAuthMiddleware,
    GraphQLErrorMiddleware,
    GraphQLRateLimitMiddleware,
)
from pynomaly.presentation.api.graphql.schema import schema


def create_graphql_app(container: Container) -> GraphQLRouter:
    """Create GraphQL application with all middleware and configuration.
    
    Args:
        container: Dependency injection container
        
    Returns:
        Configured GraphQL router
    """
    settings = container.config()
    
    # Create GraphQL router with context
    graphql_app = GraphQLRouter(
        schema,
        context_getter=get_graphql_context,
        graphiql=settings.api.docs_enabled,  # Enable GraphiQL in development
        path="/graphql"
    )
    
    # Add middleware layers
    graphql_app.add_middleware(GraphQLErrorMiddleware)
    graphql_app.add_middleware(GraphQLAuthMiddleware, container=container)
    graphql_app.add_middleware(GraphQLRateLimitMiddleware, settings=settings)
    
    return graphql_app


def integrate_graphql_with_fastapi(app: FastAPI, container: Container) -> None:
    """Integrate GraphQL with existing FastAPI application.
    
    Args:
        app: FastAPI application instance
        container: Dependency injection container
    """
    settings = container.config()
    
    # Create GraphQL router
    graphql_router = create_graphql_app(container)
    
    # Include GraphQL router with API versioning
    app.include_router(
        graphql_router,
        prefix="/api/v1",
        tags=["graphql"]
    )
    
    # Add GraphQL introspection endpoint for schema exploration
    @app.get("/api/v1/graphql/schema", tags=["graphql"])
    async def get_graphql_schema():
        """Get GraphQL schema definition."""
        from strawberry import export_schema
        return {
            "schema": export_schema(schema),
            "version": "1.0.0",
            "introspection_enabled": settings.api.docs_enabled
        }
    
    # Add GraphQL playground endpoint (development only)
    if settings.api.docs_enabled:
        @app.get("/api/v1/graphql/playground", tags=["graphql"])
        async def graphql_playground():
            """GraphQL Playground for development."""
            return {
                "message": "GraphQL Playground available at /api/v1/graphql",
                "introspection": "/api/v1/graphql/schema",
                "documentation": "Use GraphiQL interface at /api/v1/graphql"
            }