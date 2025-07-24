"""Health check endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..infrastructure.monitoring import get_health_checker

router = APIRouter()
health_checker = get_health_checker()


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="anomaly-detection-api",
        version="0.3.0",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=None  # Will be calculated in main app
    )


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes/container orchestration."""
    try:
        # Check critical components only
        critical_checks = ["algorithms"]
        
        results = {}
        overall_ready = True
        
        for check_name in critical_checks:
            result = await health_checker.run_check(check_name)
            if result:
                results[check_name] = {
                    "status": result.status.value,
                    "message": result.message
                }
                if result.status.value not in ["healthy"]:
                    overall_ready = False
            else:
                results[check_name] = {
                    "status": "unknown",
                    "message": "Check not found"
                }
                overall_ready = False
        
        return {
            "ready": overall_ready,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes/container orchestration."""
    try:
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "alive": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }