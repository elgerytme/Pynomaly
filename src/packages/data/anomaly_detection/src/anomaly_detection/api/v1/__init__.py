"""API v1 endpoints."""

from fastapi import APIRouter

from . import detection, models, streaming, explainability, workers, reports, enhanced_streaming, health
# Temporarily disabled analytics import due to pydantic forward reference issues
# from . import analytics

api_router = APIRouter()
api_router.include_router(detection.router, prefix="/detection", tags=["detection"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(streaming.router, prefix="/streaming", tags=["streaming"])
api_router.include_router(explainability.router, prefix="/explainability", tags=["explainability"])
api_router.include_router(workers.router, prefix="/workers", tags=["workers"])
api_router.include_router(reports.router, tags=["reports"])
api_router.include_router(enhanced_streaming.router, prefix="/enhanced-streaming", tags=["enhanced-streaming"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
# api_router.include_router(analytics.router, tags=["analytics"])