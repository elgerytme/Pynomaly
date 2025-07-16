"""
API Versioning Strategy for Pynomaly

This module implements a comprehensive API versioning strategy that supports:
- Semantic versioning (v1, v2, etc.)
- Backward compatibility
- Deprecation notices
- Version-specific features
- Content negotiation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIVersionStatus(str, Enum):
    """API version status enumeration."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    PREVIEW = "preview"


class VersioningStrategy(str, Enum):
    """API versioning strategy enumeration."""
    URL_PATH = "url_path"  # /api/v1/endpoint
    HEADER = "header"      # API-Version: v1
    QUERY = "query"        # ?version=v1
    MEDIA_TYPE = "media_type"  # Accept: application/vnd.monorepo.v1+json


@dataclass
class APIVersionInfo:
    """API version information."""
    version: str
    status: APIVersionStatus
    release_date: datetime
    deprecation_date: datetime | None = None
    sunset_date: datetime | None = None
    supported_until: datetime | None = None
    features: set[str] = None
    breaking_changes: list[str] = None
    migration_guide: str | None = None

    def __post_init__(self):
        """Initialize default values."""
        if self.features is None:
            self.features = set()
        if self.breaking_changes is None:
            self.breaking_changes = []


class APIVersionManager:
    """Manages API versions and compatibility."""

    def __init__(self):
        """Initialize version manager."""
        self.versions: dict[str, APIVersionInfo] = {}
        self.current_version = "v1"
        self.supported_versions = {"v1"}
        self.default_strategy = VersioningStrategy.URL_PATH

        # Initialize default versions
        self._initialize_default_versions()

    def _initialize_default_versions(self) -> None:
        """Initialize default API versions."""
        # Version 1.0 - Current stable version
        self.versions["v1"] = APIVersionInfo(
            version="v1",
            status=APIVersionStatus.ACTIVE,
            release_date=datetime(2024, 1, 1),
            features={
                "anomaly_detection",
                "model_training",
                "health_monitoring",
                "authentication",
                "basic_mlops"
            },
            breaking_changes=[],
            migration_guide="https://docs.monorepo.com/migration/v1"
        )

        # Version 2.0 - Preview version with new features
        self.versions["v2"] = APIVersionInfo(
            version="v2",
            status=APIVersionStatus.PREVIEW,
            release_date=datetime(2024, 6, 1),
            features={
                "anomaly_detection",
                "model_training",
                "health_monitoring",
                "authentication",
                "advanced_mlops",
                "multi_tenancy",
                "advanced_analytics",
                "streaming_detection",
                "ensemble_methods"
            },
            breaking_changes=[
                "Authentication endpoint response format changed",
                "Detection response includes additional metadata",
                "Training parameters validation is stricter"
            ],
            migration_guide="https://docs.monorepo.com/migration/v2"
        )

    def register_version(self, version_info: APIVersionInfo) -> None:
        """Register a new API version."""
        self.versions[version_info.version] = version_info
        logger.info(f"Registered API version: {version_info.version}")

    def get_version_info(self, version: str) -> APIVersionInfo | None:
        """Get information about a specific version."""
        return self.versions.get(version)

    def get_supported_versions(self) -> set[str]:
        """Get all supported versions."""
        return {
            version for version, info in self.versions.items()
            if info.status in [APIVersionStatus.ACTIVE, APIVersionStatus.DEPRECATED]
        }

    def get_active_versions(self) -> set[str]:
        """Get active versions."""
        return {
            version for version, info in self.versions.items()
            if info.status == APIVersionStatus.ACTIVE
        }

    def is_version_supported(self, version: str) -> bool:
        """Check if a version is supported."""
        return version in self.get_supported_versions()

    def deprecate_version(self, version: str, deprecation_date: datetime = None,
                         sunset_date: datetime = None) -> None:
        """Deprecate a version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        if deprecation_date is None:
            deprecation_date = datetime.now()

        if sunset_date is None:
            sunset_date = deprecation_date + timedelta(days=365)  # 1 year

        version_info = self.versions[version]
        version_info.status = APIVersionStatus.DEPRECATED
        version_info.deprecation_date = deprecation_date
        version_info.sunset_date = sunset_date

        logger.warning(f"Version {version} deprecated. Sunset date: {sunset_date}")

    def get_version_compatibility(self, from_version: str, to_version: str) -> dict[str, Any]:
        """Get compatibility information between versions."""
        from_info = self.get_version_info(from_version)
        to_info = self.get_version_info(to_version)

        if not from_info or not to_info:
            return {"compatible": False, "reason": "Version not found"}

        # Check feature compatibility
        missing_features = from_info.features - to_info.features
        new_features = to_info.features - from_info.features

        return {
            "compatible": len(missing_features) == 0,
            "missing_features": list(missing_features),
            "new_features": list(new_features),
            "breaking_changes": to_info.breaking_changes,
            "migration_guide": to_info.migration_guide
        }


# Response models for version information
class APIVersionResponse(BaseModel):
    """API version information response."""
    version: str = Field(..., description="API version")
    status: APIVersionStatus = Field(..., description="Version status")
    release_date: datetime = Field(..., description="Version release date")
    deprecation_date: datetime | None = Field(None, description="Version deprecation date")
    sunset_date: datetime | None = Field(None, description="Version sunset date")
    features: list[str] = Field(..., description="Available features")
    breaking_changes: list[str] = Field(..., description="Breaking changes from previous version")
    migration_guide: str | None = Field(None, description="Migration guide URL")


class APIVersionsResponse(BaseModel):
    """API versions listing response."""
    current_version: str = Field(..., description="Current API version")
    supported_versions: list[str] = Field(..., description="Supported API versions")
    versions: list[APIVersionResponse] = Field(..., description="Detailed version information")


class APICompatibilityResponse(BaseModel):
    """API compatibility information response."""
    from_version: str = Field(..., description="Source version")
    to_version: str = Field(..., description="Target version")
    compatible: bool = Field(..., description="Whether versions are compatible")
    missing_features: list[str] = Field(..., description="Features missing in target version")
    new_features: list[str] = Field(..., description="New features in target version")
    breaking_changes: list[str] = Field(..., description="Breaking changes")
    migration_guide: str | None = Field(None, description="Migration guide URL")


# Global version manager instance
version_manager = APIVersionManager()


# Dependency for version extraction
def get_api_version(request: Request) -> str:
    """Get API version from request."""
    middleware = APIVersionMiddleware(version_manager)
    version = middleware.extract_version(request)
    middleware.validate_version(version)
    return version


class APIVersionMiddleware:
    """Middleware for API versioning."""

    def __init__(self, version_manager: APIVersionManager):
        """Initialize middleware."""
        self.version_manager = version_manager

    def extract_version(self, request: Request) -> str:
        """Extract version from request."""
        # Try URL path first (/api/v1/...)
        path_parts = request.url.path.split('/')
        if len(path_parts) > 2 and path_parts[2].startswith('v'):
            return path_parts[2]

        # Try header
        version = request.headers.get("API-Version")
        if version:
            return version

        # Try query parameter
        version = request.query_params.get("version")
        if version:
            return version

        # Try media type
        accept_header = request.headers.get("Accept", "")
        if "vnd.monorepo.v" in accept_header:
            # Extract version from media type like application/vnd.monorepo.v1+json
            start = accept_header.find("vnd.monorepo.v") + len("vnd.monorepo.v")
            end = accept_header.find("+", start)
            if end == -1:
                end = accept_header.find(";", start)
            if end == -1:
                end = len(accept_header)
            return f"v{accept_header[start:end]}"

        # Default to current version
        return self.version_manager.current_version

    def validate_version(self, version: str) -> None:
        """Validate that the requested version is supported."""
        if not self.version_manager.is_version_supported(version):
            supported_versions = ", ".join(sorted(self.version_manager.get_supported_versions()))
            raise HTTPException(
                status_code=400,
                detail=f"API version '{version}' is not supported. Supported versions: {supported_versions}"
            )

        # Check if version is deprecated
        version_info = self.version_manager.get_version_info(version)
        if version_info and version_info.status == APIVersionStatus.DEPRECATED:
            logger.warning(f"Client using deprecated API version: {version}")

    def add_version_headers(self, response, version: str) -> None:
        """Add version-related headers to response."""
        version_info = self.version_manager.get_version_info(version)
        if version_info:
            response.headers["API-Version"] = version
            response.headers["API-Supported-Versions"] = ", ".join(
                sorted(self.version_manager.get_supported_versions())
            )

            if version_info.status == APIVersionStatus.DEPRECATED:
                response.headers["API-Deprecation-Date"] = version_info.deprecation_date.isoformat()
                if version_info.sunset_date:
                    response.headers["API-Sunset-Date"] = version_info.sunset_date.isoformat()
                    response.headers["Warning"] = f"299 - \"API version {version} is deprecated and will be sunset on {version_info.sunset_date.date()}\""


def get_version_manager() -> APIVersionManager:
    """Get the global version manager instance."""
    return version_manager


def create_version_router() -> APIRouter:
    """Create router for version management endpoints."""
    router = APIRouter(prefix="/api/version", tags=["API Versioning"])

    @router.get("/", response_model=APIVersionsResponse)
    async def get_versions():
        """Get all API versions."""
        versions = []
        for version, info in version_manager.versions.items():
            versions.append(APIVersionResponse(
                version=info.version,
                status=info.status,
                release_date=info.release_date,
                deprecation_date=info.deprecation_date,
                sunset_date=info.sunset_date,
                features=list(info.features),
                breaking_changes=info.breaking_changes,
                migration_guide=info.migration_guide
            ))

        return APIVersionsResponse(
            current_version=version_manager.current_version,
            supported_versions=list(version_manager.get_supported_versions()),
            versions=versions
        )

    @router.get("/{version}", response_model=APIVersionResponse)
    async def get_version(version: str):
        """Get specific version information."""
        info = version_manager.get_version_info(version)
        if not info:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")

        return APIVersionResponse(
            version=info.version,
            status=info.status,
            release_date=info.release_date,
            deprecation_date=info.deprecation_date,
            sunset_date=info.sunset_date,
            features=list(info.features),
            breaking_changes=info.breaking_changes,
            migration_guide=info.migration_guide
        )

    @router.get("/{from_version}/compatibility/{to_version}", response_model=APICompatibilityResponse)
    async def get_compatibility(from_version: str, to_version: str):
        """Get compatibility information between versions."""
        compatibility = version_manager.get_version_compatibility(from_version, to_version)

        return APICompatibilityResponse(
            from_version=from_version,
            to_version=to_version,
            compatible=compatibility["compatible"],
            missing_features=compatibility.get("missing_features", []),
            new_features=compatibility.get("new_features", []),
            breaking_changes=compatibility.get("breaking_changes", []),
            migration_guide=compatibility.get("migration_guide")
        )

    return router
