"""API v1 endpoints."""

from fastapi import APIRouter

from . import detection, models, monitoring, streaming, explainability, workers, health

api_router = APIRouter()
api_router.include_router(detection.router, prefix="/detection", tags=["detection"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(streaming.router, prefix="/streaming", tags=["streaming"])
api_router.include_router(explainability.router, prefix="/explainability", tags=["explainability"])
api_router.include_router(workers.router, prefix="/workers", tags=["workers"])
api_router.include_router(health.router, prefix="/health", tags=["health"])