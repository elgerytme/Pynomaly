"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from pynomaly.infrastructure.auth import init_auth, track_request_metrics
from pynomaly.infrastructure.cache import init_cache
from pynomaly.infrastructure.config import Container

# Temporarily disabled telemetry
# from pynomaly.infrastructure.monitoring import init_telemetry
from pynomaly.presentation.api.docs import api_docs, configure_openapi_docs
from pynomaly.presentation.api.endpoints import (
    admin,
    auth,
    automl,
    autonomous,
    datasets,
    detection,
    detectors,
    ensemble,
    events,
    experiments,
    explainability,
    export,
    health,
    model_lineage,
    performance,
    streaming,
)

# Enhanced AutoML endpoints
try:
    from pynomaly.presentation.api import enhanced_automl

    ENHANCED_AUTOML_AVAILABLE = True
except ImportError:
    ENHANCED_AUTOML_AVAILABLE = False

# Distributed processing endpoints removed for simplification
distributed = None
DISTRIBUTED_API_AVAILABLE = False


# Web UI mounting - resolved circular import by using late import
def _mount_web_ui_lazy(app):
    """Lazy import and mount web UI to avoid circular imports."""
    try:
        from pynomaly.presentation.web.app import mount_web_ui

        mount_web_ui(app)
        return True
    except ImportError as e:
        print(f"Warning: Could not mount web UI: {e}")
        return False


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
        # init_telemetry(settings)  # Temporarily disabled
        pass

    yield

    # Shutdown
    print("Shutting down...")

    # Cleanup services
    # Telemetry cleanup temporarily disabled
    # from pynomaly.infrastructure.monitoring import get_telemetry
    # telemetry = get_telemetry()
    # if telemetry:
    #     telemetry.shutdown()

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

    # Create app with enhanced documentation configuration
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="""
# Pynomaly - Advanced Anomaly Detection Platform

**Pynomaly** is a state-of-the-art Python anomaly detection package that provides a unified, production-ready interface for multiple anomaly detection algorithms.

## Key Features

🚀 **Multi-Algorithm Support**: Integrates PyOD, TODS, PyGOD, scikit-learn, PyTorch, TensorFlow, and JAX  
🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture  
🔒 **Enterprise Security**: JWT authentication, RBAC, audit logging, and encryption  
⚡ **High Performance**: Distributed processing, caching, and performance optimization  
📊 **Advanced Analytics**: AutoML, explainability, and comprehensive visualization  
🌐 **Progressive Web App**: Modern UI with offline capabilities  
📈 **Production Ready**: Monitoring, observability, and enterprise deployment features

## Quick Start

1. **Authenticate**: Use `/auth/login` to get a JWT token
2. **Upload Data**: Use `/datasets/upload` to upload your dataset  
3. **Create Detector**: Use `/detectors/create` to configure an anomaly detector
4. **Train Model**: Use `/detection/train` to train the detector
5. **Detect Anomalies**: Use `/detection/predict` to find anomalies

## Documentation

- **Interactive API Explorer**: `/docs/swagger`
- **API Reference**: `/docs/redoc`
- **Postman Collection**: `/docs/postman`
- **SDK Information**: `/docs/sdk-info`

## Support

- **GitHub**: [https://github.com/pynomaly/pynomaly](https://github.com/pynomaly/pynomaly)
- **Documentation**: [https://pynomaly.readthedocs.io](https://pynomaly.readthedocs.io)
- **Issues**: [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)
        """,
        docs_url="/api/docs" if settings.docs_enabled else None,
        redoc_url="/api/redoc" if settings.docs_enabled else None,
        openapi_url="/api/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
        contact={
            "name": "Pynomaly Team",
            "url": "https://github.com/pynomaly/pynomaly",
            "email": "team@pynomaly.io",
        },
        license_info={
            "name": "MIT",
            "url": "https://github.com/pynomaly/pynomaly/blob/main/LICENSE",
        },
        terms_of_service="https://pynomaly.io/terms",
    )

    # Store container in app state
    app.state.container = container

    # Configure comprehensive OpenAPI documentation
    configure_openapi_docs(app, settings)

    # Add CORS middleware
    app.add_middleware(CORSMiddleware, **settings.get_cors_config())

    # Add request tracking middleware
    app.middleware("http")(track_request_metrics)

    # Add Prometheus metrics if enabled
    if settings.monitoring.prometheus_enabled:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")

    # Include documentation router (before API routers for proper URL handling)
    app.include_router(api_docs.router, tags=["documentation"])

    # Include API routers
    app.include_router(health.router, prefix="/api/health", tags=["health"])

    app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])

    app.include_router(admin.router, prefix="/api/admin", tags=["administration"])

    app.include_router(autonomous.router, prefix="/api/autonomous", tags=["autonomous"])

    app.include_router(detectors.router, prefix="/api/detectors", tags=["detectors"])

    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])

    app.include_router(detection.router, prefix="/api/detection", tags=["detection"])

    app.include_router(automl.router, prefix="/api/automl", tags=["automl"])

    # Include enhanced AutoML router if available
    if ENHANCED_AUTOML_AVAILABLE:
        app.include_router(enhanced_automl.router, tags=["enhanced_automl"])

    app.include_router(ensemble.router, prefix="/api/ensemble", tags=["ensemble"])

    app.include_router(
        explainability.router, prefix="/api/explainability", tags=["explainability"]
    )

    app.include_router(
        experiments.router, prefix="/api/experiments", tags=["experiments"]
    )

    app.include_router(
        performance.router, prefix="/api/performance", tags=["performance"]
    )

    app.include_router(export.router, prefix="/api", tags=["export"])

    # Advanced model management endpoints
    app.include_router(model_lineage.router, prefix="/api", tags=["model_lineage"])

    # Real-time streaming and event processing endpoints
    app.include_router(streaming.router, prefix="/api", tags=["streaming"])
    app.include_router(events.router, prefix="/api", tags=["events"])

    # Distributed processing API removed for simplification

    # Mount web UI with lazy import to avoid circular dependencies
    _mount_web_ui_lazy(app)

    @app.get("/", include_in_schema=False)
    async def root():
        """API root endpoint."""
        return {
            "message": "Pynomaly API",
            "version": settings.app.version,
            "docs": "/api/docs",
            "health": "/api/health",
        }

    return app


# Create default app instance for uvicorn
app = create_app()
