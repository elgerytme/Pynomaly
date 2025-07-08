"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Optional prometheus dependency
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Instrumentator = None

from pynomaly.infrastructure.auth import init_auth, track_request_metrics
from pynomaly.infrastructure.cache import init_cache
from pynomaly.infrastructure.config import Container
from pynomaly.infrastructure.monitoring.service_initialization import (
    initialize_monitoring_service,
    shutdown_monitoring_service,
)

# Temporarily disabled telemetry
# from pynomaly.infrastructure.monitoring import init_telemetry
from pynomaly.presentation.api.docs import api_docs, configure_openapi_docs
from pynomaly.presentation.api.router_factory import apply_openapi_overrides
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
    version,
)
from pynomaly.presentation.api.routers import user_management
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
    
    # Initialize external monitoring service
    try:
        monitoring_service = await initialize_monitoring_service(settings)
        app.state.monitoring_service = monitoring_service
        print(f"Initialized external monitoring service with {len(monitoring_service.providers)} providers")
        
        # Initialize dual metrics service
        from pynomaly.infrastructure.monitoring.dual_metrics_service import (
            initialize_dual_metrics_service,
        )
        from pynomaly.infrastructure.monitoring.prometheus_metrics import (
            get_metrics_service,
            initialize_metrics,
        )
        
        # Initialize Prometheus metrics if not already done
        prometheus_service = get_metrics_service()
        if prometheus_service is None and settings.monitoring.prometheus_enabled:
            prometheus_service = initialize_metrics(
                enable_default_metrics=True,
                namespace="pynomaly",
                port=settings.monitoring.prometheus_port if settings.monitoring.prometheus_enabled else None,
            )
        
        # Initialize dual metrics service
        dual_metrics_service = initialize_dual_metrics_service(
            prometheus_service=prometheus_service,
            external_service=monitoring_service,
        )
        app.state.dual_metrics_service = dual_metrics_service
        print("Initialized dual metrics service")
        
    except Exception as e:
        print(f"Warning: Failed to initialize monitoring service: {e}")
        app.state.monitoring_service = None
        app.state.dual_metrics_service = None

    yield

    # Shutdown
    print("Shutting down...")

    # Shutdown monitoring service
    if hasattr(app.state, 'monitoring_service') and app.state.monitoring_service is not None:
        try:
            await shutdown_monitoring_service(app.state.monitoring_service)
            print("Monitoring service shutdown complete")
        except Exception as e:
            print(f"Warning: Error shutting down monitoring service: {e}")

    # Clear dependencies on shutdown
    from pynomaly.infrastructure.dependencies import clear_dependencies
    clear_dependencies()

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
    # If container is provided, skip wiring to avoid import issues during testing

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

1. **Authenticate**: Use `/api/v1/auth/login` to get a JWT token
2. **Upload Data**: Use `/api/v1/datasets/upload` to upload your dataset  
3. **Create Detector**: Use `/api/v1/detectors/create` to configure an anomaly detector
4. **Train Model**: Use `/api/v1/detection/train` to train the detector
5. **Detect Anomalies**: Use `/api/v1/detection/predict` to find anomalies

## Documentation

- **Interactive API Explorer**: `/api/v1/docs`
- **API Reference**: `/api/v1/redoc`
- **Version Information**: `/api/v1/version`
- **Postman Collection**: `/docs/postman`
- **SDK Information**: `/docs/sdk-info`

## Support

- **GitHub**: [https://github.com/pynomaly/pynomaly](https://github.com/pynomaly/pynomaly)
- **Documentation**: [https://pynomaly.readthedocs.io](https://pynomaly.readthedocs.io)
- **Issues**: [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)
        """,
        docs_url="/api/v1/docs" if settings.docs_enabled else None,
        redoc_url="/api/v1/redoc" if settings.docs_enabled else None,
        openapi_url="/api/v1/openapi.json" if settings.docs_enabled else None,
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

    # Apply dependency overrides to resolve circular dependency issues
    # This enables OpenAPI generation by replacing complex Annotated[Depends(...)] patterns
    apply_openapi_overrides(app)
    
    # Configure OpenAPI documentation
    configure_openapi_docs(app, settings)

    # Add CORS middleware
    app.add_middleware(CORSMiddleware, **settings.get_cors_config())

    # Add request tracking middleware
    app.middleware("http")(track_request_metrics)

    # Add Prometheus metrics if enabled and available
    if settings.monitoring.prometheus_enabled and PROMETHEUS_AVAILABLE:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    elif settings.monitoring.prometheus_enabled and not PROMETHEUS_AVAILABLE:
        print("Warning: Prometheus metrics requested but prometheus-fastapi-instrumentator not available")

    # Include documentation router (before API routers for proper URL handling)
    app.include_router(api_docs.router, tags=["documentation"])

    # Include API routers with v1 versioning
    # URL Strategy: All API endpoints use "/api/v1" prefix for versioned REST API
    # Web UI is served at root "/" path, API at "/api/v1" - see docs/url_scheme.md
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])

    app.include_router(user_management.router, prefix="/api/v1", tags=["user_management"])

    app.include_router(admin.router, prefix="/api/v1/admin", tags=["administration"])

    app.include_router(autonomous.router, prefix="/api/v1/autonomous", tags=["autonomous"])

    app.include_router(detectors.router, prefix="/api/v1/detectors", tags=["detectors"])

    app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])

    app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"])

    app.include_router(automl.router, prefix="/api/v1/automl", tags=["automl"])

    # Include enhanced AutoML router if available
    if ENHANCED_AUTOML_AVAILABLE:
        app.include_router(enhanced_automl.router, prefix="/api/v1", tags=["enhanced_automl"])

    app.include_router(ensemble.router, prefix="/api/v1/ensemble", tags=["ensemble"])

    app.include_router(
        explainability.router, prefix="/api/v1/explainability", tags=["explainability"]
    )

    app.include_router(
        experiments.router, prefix="/api/v1/experiments", tags=["experiments"]
    )

    # Include version endpoint
    app.include_router(version.router, prefix="/api/v1", tags=["version"])

    app.include_router(
        performance.router, prefix="/api/v1/performance", tags=["performance"]
    )

    app.include_router(export.router, prefix="/api/v1", tags=["export"])

    # Advanced model management endpoints
    app.include_router(model_lineage.router, prefix="/api/v1", tags=["model_lineage"])

    # Real-time streaming and event processing endpoints
    app.include_router(streaming.router, prefix="/api/v1", tags=["streaming"])
    app.include_router(events.router, prefix="/api/v1", tags=["events"])

    # Distributed processing API removed for simplification

    # Mount web UI with lazy import to avoid circular dependencies
    # URL Strategy: Web UI served at root "/" path for clean user experience
    # Separates UI from API routes ("/api/v1") - see docs/url_scheme.md
    _mount_web_ui_lazy(app)

    @app.get("/", include_in_schema=False)
    async def root():
        """API root endpoint."""
        return {
            "message": "Pynomaly API",
            "version": settings.app.version,
            "api_version": "v1",
            "docs": "/api/v1/docs",
            "health": "/api/v1/health",
            "version_info": "/api/v1/version",
        }

    return app


# Create default app instance for uvicorn - commented out to avoid import issues
# app = create_app()
