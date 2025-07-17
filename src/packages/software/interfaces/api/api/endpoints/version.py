"""
Generic Version Endpoint

Provides version information for software applications.
"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/version")
async def get_version() -> Dict[str, Any]:
    """Get application version information"""
    return {
        "version": "0.1.0",
        "service": "software-api",
        "build": "latest"
    }