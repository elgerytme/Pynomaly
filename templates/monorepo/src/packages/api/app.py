"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from packages.infrastructure.config.settings import settings
from .endpoints import health, users
from .middleware.exception_handler import add_exception_handlers
from .middleware.logging import LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    
    yield
    
    # Shutdown
    print(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A comprehensive application built with Clean Architecture",
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else [settings.host],
    )
    
    app.add_middleware(LoggingMiddleware)
    
    # Add exception handlers
    add_exception_handlers(app)
    
    # Add routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    
    return app


def main() -> None:
    """Main entry point for running the API server."""
    import uvicorn
    
    uvicorn.run(
        "packages.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    main()