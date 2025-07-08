#!/usr/bin/env python3
"""Simple app runner that bypasses circular imports for OpenAPI generation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pynomaly.infrastructure.config import Container, create_container
from pynomaly.presentation.api.docs import configure_openapi_docs
from pynomaly.presentation.api.router_factory import apply_openapi_overrides

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Simplified lifespan manager for OpenAPI generation."""
    # Startup
    print("Starting Pynomaly API...")

    # Apply dependency overrides to fix forward reference issues
    apply_openapi_overrides(app)

    yield

    # Shutdown
    print("Shutting down...")


def create_minimal_app(container: Container | None = None) -> FastAPI:
    """Create a minimal FastAPI app for OpenAPI generation.

    Args:
        container: Optional DI container

    Returns:
        Configured FastAPI application
    """
    if container is None:
        container = create_container()

    settings = container.config()

    # Create minimal app
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Pynomaly - Advanced Anomaly Detection Platform",
        docs_url="/api/v1/docs" if settings.docs_enabled else None,
        redoc_url="/api/v1/redoc" if settings.docs_enabled else None,
        openapi_url="/api/v1/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )

    # Store container in app state
    app.state.container = container

    # Apply dependency overrides
    apply_openapi_overrides(app)

    # Configure OpenAPI documentation
    configure_openapi_docs(app, settings)

    # Add CORS middleware
    app.add_middleware(CORSMiddleware, **settings.get_cors_config())

    # Add a simple health endpoint
    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "pynomaly-api"}

    # Add root endpoint
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "message": "Pynomaly API",
            "version": settings.app.version,
            "docs": "/api/v1/docs",
            "health": "/api/health",
        }

    return app


# Create app instance for uvicorn
app = create_minimal_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
