"""
Main server application for the MLOps Marketplace.

Combines all API endpoints, middleware, and services into a single
FastAPI application for the marketplace platform.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import structlog

from mlops_marketplace.infrastructure.api.gateway import APIGateway, APIGatewayConfig
from mlops_marketplace.infrastructure.api.rate_limiter import RateLimiter
from mlops_marketplace.infrastructure.persistence.redis_cache import RedisCache
from mlops_marketplace.infrastructure.monitoring import (
    PrometheusMetrics,
    StructlogLogger,
    OpenTelemetryTracer,
)
from mlops_marketplace.presentation.api.routes import (
    solutions_router,
    deployments_router,
    subscriptions_router,
    reviews_router,
    analytics_router,
    admin_router,
    auth_router,
    users_router,
)
from mlops_marketplace.application.container import ApplicationContainer


logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting MLOps Marketplace server")
    
    # Initialize application container
    container = ApplicationContainer()
    await container.init_resources()
    app.state.container = container
    
    # Start background services
    await container.start_background_services()
    
    yield
    
    # Shutdown
    logger.info("Shutting down MLOps Marketplace server")
    await container.cleanup_resources()


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or {}
    
    # Configure logging
    StructlogLogger.configure(
        log_level=config.get("log_level", "INFO"),
        json_logs=config.get("json_logs", True),
        development=config.get("development", False),
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="MLOps Marketplace API",
        description="Enterprise MLOps Marketplace and Ecosystem Platform",
        version="1.0.0",
        docs_url="/docs" if config.get("enable_docs", True) else None,
        redoc_url="/redoc" if config.get("enable_docs", True) else None,
        openapi_url="/openapi.json" if config.get("enable_docs", True) else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Mount static files
    if config.get("static_files_directory"):
        app.mount(
            "/static",
            StaticFiles(directory=config["static_files_directory"]),
            name="static"
        )
    
    # Include API routers
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(solutions_router, prefix="/api/v1/solutions", tags=["Solutions"])
    app.include_router(deployments_router, prefix="/api/v1/deployments", tags=["Deployments"])
    app.include_router(subscriptions_router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])
    app.include_router(reviews_router, prefix="/api/v1/reviews", tags=["Reviews"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["Administration"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        container = getattr(app.state, "container", None)
        if not container:
            raise HTTPException(status_code=503, detail="Application not ready")
        
        health_status = await container.check_health()
        
        if not health_status["healthy"]:
            raise HTTPException(
                status_code=503,
                detail="Service unhealthy",
                headers={"X-Health-Details": str(health_status)}
            )
        
        return {
            "status": "healthy",
            "timestamp": health_status["timestamp"],
            "version": "1.0.0",
            "services": health_status["services"],
        }
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Metrics endpoint for Prometheus."""
        container = getattr(app.state, "container", None)
        if not container or not container.metrics:
            raise HTTPException(status_code=404, detail="Metrics not available")
        
        return container.metrics.generate_metrics()
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "MLOps Marketplace API",
            "version": "1.0.0",
            "description": "Enterprise MLOps Marketplace and Ecosystem Platform",
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            path=request.url.path,
            method=request.method,
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "path": request.url.path,
            }
        )
    
    return app


def main():
    """Main entry point for the server."""
    # Load configuration from environment
    config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8000")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "development": os.getenv("ENVIRONMENT", "production") == "development",
        "reload": os.getenv("RELOAD", "false").lower() == "true",
        "workers": int(os.getenv("WORKERS", "1")),
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "enable_docs": os.getenv("ENABLE_DOCS", "true").lower() == "true",
        "static_files_directory": os.getenv("STATIC_FILES_DIRECTORY"),
        "json_logs": os.getenv("JSON_LOGS", "true").lower() == "true",
    }
    
    # Create application
    app = create_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"].lower(),
        reload=config["reload"],
        workers=config["workers"] if not config["reload"] else 1,
        access_log=True,
        use_colors=not config["json_logs"],
    )


if __name__ == "__main__":
    main()