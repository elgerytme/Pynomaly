"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from pynomaly.infrastructure.security.rate_limiting_middleware import (
    RateLimitMiddleware,
)
from pynomaly.presentation.api.middleware import SecurityHeadersMiddleware

# Optional prometheus dependency
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Instrumentator = None

# Production monitoring integration
try:
    from pynomaly.infrastructure.monitoring.fastapi_monitoring_middleware import (
        setup_monitoring_middleware,
    )
    from pynomaly.infrastructure.monitoring.production_monitoring_integration import (
        create_production_monitoring,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from pynomaly.infrastructure.auth import init_auth, track_request_metrics
from pynomaly.infrastructure.cache import init_cache
from pynomaly.infrastructure.config import Container

# Temporarily disabled telemetry
# from pynomaly.infrastructure.monitoring import init_telemetry
from pynomaly.presentation.api.docs import api_docs, configure_openapi_docs
from pynomaly.presentation.api.endpoints import (
    auth,
    events,
    export,
    frontend_support,
    health,
    model_lineage,
    performance,
    security_management,
    streaming,
    version,
)
from pynomaly.presentation.api.router_factory import apply_openapi_overrides

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
    if settings.storage.cache_enabled:
        init_cache(settings)

    if settings.security.auth_enabled:
        init_auth(settings)

    # Initialize production monitoring
    if MONITORING_AVAILABLE and getattr(settings, "monitoring_enabled", False):
        try:
            monitoring = create_production_monitoring()
            await monitoring.initialize()
            app.state.monitoring = monitoring

            # Set up monitoring middleware now that monitoring is initialized
            setup_monitoring_middleware(app, monitoring)

            print("✅ Production monitoring initialized and middleware configured")
        except Exception as e:
            print(f"❌ Failed to initialize monitoring: {e}")

    if settings.monitoring.metrics_enabled or settings.monitoring.tracing_enabled:
        # init_telemetry(settings)  # Temporarily disabled
        pass

    yield

    # Shutdown
    print("Shutting down...")

    # Cleanup monitoring
    if hasattr(app.state, "monitoring"):
        try:
            await app.state.monitoring.shutdown()
            print("✅ Production monitoring shutdown complete")
        except Exception as e:
            print(f"❌ Error shutting down monitoring: {e}")

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
        docs_url="/api/v1/docs" if settings.api.docs_enabled else None,
        redoc_url="/api/v1/redoc" if settings.api.docs_enabled else None,
        openapi_url="/api/v1/openapi.json" if settings.api.docs_enabled else None,
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

    # Add security headers middleware first
    app.add_middleware(SecurityHeadersMiddleware)

    # Add rate limiting middleware (after security headers, before CORS)
    app.add_middleware(RateLimitMiddleware, settings=settings)

    # Add CORS middleware
    app.add_middleware(CORSMiddleware, **settings.get_cors_config())

    # Add session middleware for CSRF protection
    app.add_middleware(SessionMiddleware, secret_key=settings.security.secret_key)

    # Add request tracking middleware
    app.middleware("http")(track_request_metrics)

    # Setup comprehensive security monitoring for API
    try:
        from pynomaly.presentation.web.security.security_monitor import (
            setup_security_monitoring,
        )

        setup_security_monitoring(
            app,
            config={
                "monitoring_enabled": True,
                "rate_limit_per_minute": 1000,  # Higher limit for API
                "auto_block_threshold": 10,
                "block_duration": 1800,
            },
        )
        print("✅ API security monitoring enabled")
    except Exception as e:
        print(f"❌ API security monitoring setup failed: {e}")

    # Add production monitoring middleware if available
    if MONITORING_AVAILABLE and getattr(settings, "monitoring_enabled", False):
        try:
            # Note: Monitoring integration will be set up in lifespan
            # Middleware setup is deferred until monitoring is initialized
            pass
        except Exception as e:
            print(f"Warning: Could not set up monitoring middleware: {e}")

    # Add Prometheus metrics if enabled and available
    if settings.monitoring.prometheus_enabled and PROMETHEUS_AVAILABLE:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    elif settings.monitoring.prometheus_enabled and not PROMETHEUS_AVAILABLE:
        print(
            "Warning: Prometheus metrics requested but prometheus-fastapi-instrumentator not available"
        )

    # Include documentation router (before API routers for proper URL handling)
    app.include_router(api_docs.router, tags=["documentation"])

    # Include API routers with v1 versioning
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])

    from pynomaly.presentation.api.endpoints import mfa
    app.include_router(mfa.router, prefix="/api/v1", tags=["mfa"])

    # app.include_router(
    #     user_management.router, prefix="/api/v1", tags=["user_management"]
    # )  # Temporarily disabled

    # app.include_router(admin.router, prefix="/api/v1/admin", tags=["administration"])  # Temporarily disabled

    # app.include_router(
    #     autonomous.router, prefix="/api/v1/autonomous", tags=["autonomous"]
    # )  # Temporarily disabled

    # app.include_router(detectors.router, prefix="/api/v1/detectors", tags=["detectors"])  # Temporarily disabled

    # app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])  # Temporarily disabled

    # app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"])  # Temporarily disabled

    # app.include_router(automl.router, prefix="/api/v1/automl", tags=["automl"])  # Temporarily disabled

    # Include enhanced AutoML router if available
    # if ENHANCED_AUTOML_AVAILABLE:
    #     app.include_router(
    #         enhanced_automl.router, prefix="/api/v1", tags=["enhanced_automl"]
    #     )  # Temporarily disabled

    from pynomaly.presentation.api.endpoints import ensemble, explainability
    app.include_router(ensemble.router, prefix="/api/v1/ensemble", tags=["ensemble"])
    app.include_router(
        explainability.router, prefix="/api/v1/explainability", tags=["explainability"]
    )

    # app.include_router(
    #     experiments.router, prefix="/api/v1/experiments", tags=["experiments"]
    # )  # Temporarily disabled

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

    # Frontend support endpoints for web UI utilities
    app.include_router(frontend_support.router, tags=["frontend_support"])

    # Security management endpoints
    app.include_router(security_management.router, tags=["security"])

    # Distributed processing API removed for simplification

    # Mount web UI with lazy import to avoid circular dependencies
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


# Create default app instance for uvicorn
# app = create_app()  # Commented out to prevent immediate creation during import
