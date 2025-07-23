"""Main FastAPI application with HTMX and Tailwind CSS for Anomaly Detection."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import htmx, pages, analytics
from ..infrastructure.config.settings import get_settings
from ..infrastructure.logging import setup_logging

logger = structlog.get_logger()
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    setup_logging()
    logger.info("Starting up Anomaly Detection Web App", version=app.version)
    
    # Initialize any startup tasks here
    logger.info("Web services initialized successfully")
    
    yield
    
    logger.info("Shutting down Anomaly Detection Web App")


def create_web_app() -> FastAPI:
    """Create FastAPI web application instance."""
    app = FastAPI(
        title="Anomaly Detection Dashboard",
        description="Interactive web interface for ML-based anomaly detection",
        version="0.3.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    # Include routers
    app.include_router(pages.router)
    app.include_router(htmx.router, prefix="/htmx")
    app.include_router(analytics.router, prefix="/htmx/analytics")

    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """404 error handler."""
        return templates.TemplateResponse(
            "pages/404.html",
            {"request": request},
            status_code=404,
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        """500 error handler."""
        return templates.TemplateResponse(
            "pages/500.html",
            {"request": request},
            status_code=500,
        )

    return app


web_app = create_web_app()


def main() -> None:
    """Run the web application."""
    import uvicorn
    uvicorn.run(
        "anomaly_detection.web.main:web_app",
        host=settings.api.host,
        port=settings.api.port + 1,  # Use different port from API
        reload=settings.debug,
        log_level=settings.logging.level.lower()
    )


if __name__ == "__main__":
    main()