"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

from packages.infrastructure.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint."""
    # Add checks for database, cache, etc.
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": "healthy",
            "cache": "healthy",
        },
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check endpoint."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }