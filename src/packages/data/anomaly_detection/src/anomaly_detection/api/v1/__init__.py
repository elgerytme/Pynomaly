"""API v1 endpoints."""

from fastapi import APIRouter

from . import detection, models, monitoring

api_router = APIRouter()
api_router.include_router(detection.router, prefix="/detection", tags=["detection"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])