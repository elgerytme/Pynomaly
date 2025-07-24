"""Main FastAPI application for Anomaly Detection API."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import health
from .api.v1 import api_router
from .infrastructure.config.settings import get_settings
from .infrastructure.logging import setup_logging

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    setup_logging()
    logger.info("Starting up Anomaly Detection API", version=app.version)
    
    # Initialize any startup tasks here
    logger.info("Services initialized successfully")
    
    yield
    
    logger.info("Shutting down Anomaly Detection API")
    # Cleanup resources if needed


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Anomaly Detection API",
        description="Production-ready API for ML-based anomaly detection with ensemble methods and model management",
        version="0.3.0",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()


def main() -> None:
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower()
    )


if __name__ == "__main__":
    main()