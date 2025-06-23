"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    checks: Dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def health_check(
    container: Container = Depends(get_container)
) -> HealthResponse:
    """Check application health."""
    settings = container.config()
    
    # Check various components
    checks = {
        "database": "ok",  # TODO: Implement actual DB check
        "storage": "ok",
        "cache": "ok" if settings.cache_enabled else "disabled",
    }
    
    # Check repositories
    try:
        detector_count = container.detector_repository().count()
        dataset_count = container.dataset_repository().count()
        checks["repositories"] = {
            "detectors": detector_count,
            "datasets": dataset_count,
            "status": "ok"
        }
    except Exception as e:
        checks["repositories"] = {"status": "error", "error": str(e)}
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.version,
        checks=checks
    )


@router.get("/ready")
async def readiness_check(
    container: Container = Depends(get_container)
) -> Dict[str, str]:
    """Kubernetes readiness probe."""
    # Check if app is ready to serve requests
    try:
        # Try to access repositories
        container.detector_repository().count()
        container.dataset_repository().count()
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe."""
    # Simple check that app is running
    return {"status": "alive"}