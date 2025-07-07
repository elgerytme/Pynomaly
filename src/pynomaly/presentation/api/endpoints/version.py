"""Version information endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from pynomaly.presentation.api.versioning import get_version_manager

router = APIRouter()


class VersionInfo(BaseModel):
    """API version information response model."""
    
    supported_versions: List[str]
    default_version: str
    deprecated_versions: List[str]
    current_version: str = "v1"


@router.get("/version", response_model=VersionInfo)
async def get_api_version_info() -> VersionInfo:
    """Get API version information.
    
    Returns information about supported, deprecated, and current API versions.
    """
    version_manager = get_version_manager()
    version_info = version_manager.get_version_info()
    
    return VersionInfo(
        supported_versions=version_info["supported_versions"],
        default_version=version_info["default_version"],
        deprecated_versions=version_info["deprecated_versions"],
        current_version="v1"
    )


@router.get("/")
async def api_root():
    """API root endpoint with version information."""
    version_manager = get_version_manager()
    version_info = version_manager.get_version_info()
    
    return {
        "message": "Pynomaly API",
        "version_info": version_info,
        "documentation": {
            "interactive": "/api/v1/docs",
            "redoc": "/api/v1/redoc",
            "openapi": "/api/v1/openapi.json"
        },
        "endpoints": {
            "health": "/api/v1/health",
            "auth": "/api/v1/auth",
            "datasets": "/api/v1/datasets",
            "detectors": "/api/v1/detectors",
            "detection": "/api/v1/detection"
        }
    }
