"""Data Science FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator

from .infrastructure.di.container import Container
from .presentation.api import experiments, features, metrics

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Data Science API server")
    # Initialize services, database connections, etc.
    yield
    logger.info("Shutting down Data Science API server")
    # Cleanup resources


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Data Science API",
        description="API for data science experiment management and feature validation",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Include routers
    app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
    app.include_router(features.router, prefix="/api/v1/features", tags=["features"])  
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-science"}


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "data_science.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()