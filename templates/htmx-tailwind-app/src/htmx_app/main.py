"""FastAPI application with HTMX and Tailwind CSS."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from htmx_app.api import pages, components, forms, data
from htmx_app.core.config import get_settings

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="HTMX + Tailwind CSS App",
    description="Modern web application with HTMX and Tailwind CSS",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
)

# Setup static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Make templates globally available
app.state.templates = templates

# Include routers
app.include_router(pages.router, tags=["pages"])
app.include_router(components.router, prefix="/api/components", tags=["components"])
app.include_router(forms.router, prefix="/api/forms", tags=["forms"])
app.include_router(data.router, prefix="/api/data", tags=["data"])


@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Root endpoint - redirect to home page."""
    return templates.TemplateResponse(
        "pages/home.html",
        {"request": request, "title": "Home"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "htmx_app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )