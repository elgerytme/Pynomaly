"""Minimal analytics API for testing."""

from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/health")
async def analytics_health() -> Dict[str, Any]:
    """Basic analytics health check."""
    return {"status": "ok", "service": "analytics"}