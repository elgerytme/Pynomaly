"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from pynomaly.infrastructure.config import Container, Settings
from pynomaly.infrastructure.auth import init_auth, track_request_metrics
from pynomaly.infrastructure.cache import init_cache
from pynomaly.infrastructure.monitoring import init_telemetry
from pynomaly.presentation.api.endpoints import (
    auth,
    datasets,
    detectors,
    detection,
    experiments,
    health
)
from pynomaly.presentation.web.app import mount_web_ui


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    container = app.state.container
    settings = container.config()
    
    print(f"Starting {settings.app.name} v{settings.app.version}")
    
    # Initialize services
    if settings.cache_enabled:
        init_cache(settings)
    
    if settings.auth_enabled:
        init_auth(settings)
    
    if settings.monitoring.metrics_enabled or settings.monitoring.tracing_enabled:
        init_telemetry(settings)
    
    yield
    
    # Shutdown
    print("Shutting down...")
    
    # Cleanup services
    from pynomaly.infrastructure.monitoring import get_telemetry
    telemetry = get_telemetry()
    if telemetry:
        telemetry.shutdown()
    
    from pynomaly.infrastructure.cache import get_cache
    cache = get_cache()
    if cache:
        cache.close()


def create_app(container: Container | None = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        container: Optional DI container (creates default if not provided)
        
    Returns:
        Configured FastAPI application
    """
    if container is None:
        from pynomaly.infrastructure.config import create_container
        container = create_container()
    
    settings = container.config()
    
    # Create app
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="State-of-the-art anomaly detection platform",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Store container in app state
    app.state.container = container
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        **settings.get_cors_config()
    )
    
    # Add request tracking middleware
    app.middleware("http")(track_request_metrics)
    
    # Add Prometheus metrics if enabled
    if settings.monitoring.prometheus_enabled:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    
    # Include API routers
    app.include_router(
        health.router,
        prefix="/api/health",
        tags=["health"]
    )
    
    app.include_router(
        auth.router,
        prefix="/api/auth",
        tags=["authentication"]
    )
    
    app.include_router(
        detectors.router,
        prefix="/api/detectors",
        tags=["detectors"]
    )
    
    app.include_router(
        datasets.router,
        prefix="/api/datasets",
        tags=["datasets"]
    )
    
    app.include_router(
        detection.router,
        prefix="/api/detection",
        tags=["detection"]
    )
    
    app.include_router(
        experiments.router,
        prefix="/api/experiments",
        tags=["experiments"]
    )
    
    # Mount web UI
    mount_web_ui(app)
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect to web UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/web/")
    
    return app


# Create default app instance for uvicorn
app = create_app()