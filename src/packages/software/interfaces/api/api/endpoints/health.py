"""
Generic Health Check Endpoint

Provides basic health check functionality for software applications.
"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "software-api",
        "version": "0.1.0"
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "service": "software-api",
        "version": "0.1.0"
    }