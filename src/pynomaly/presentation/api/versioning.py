"""API versioning utilities for Pynomaly API."""

from enum import Enum

from fastapi import APIRouter, FastAPI


class APIVersion(Enum):
    """Supported API versions."""

    V1 = "v1"
    # Future versions can be added here
    # V2 = "v2"


class VersionManager:
    """Manages API versions and routing."""

    def __init__(self):
        self.supported_versions = [APIVersion.V1]
        self.default_version = APIVersion.V1
        self.deprecated_versions: list[APIVersion] = []

    def get_version_prefix(self, version: APIVersion) -> str:
        """Get the URL prefix for a given API version."""
        return f"/api/{version.value}"

    def is_supported(self, version: APIVersion) -> bool:
        """Check if an API version is supported."""
        return version in self.supported_versions

    def is_deprecated(self, version: APIVersion) -> bool:
        """Check if an API version is deprecated."""
        return version in self.deprecated_versions

    def get_version_info(self) -> dict[str, any]:
        """Get information about all API versions."""
        return {
            "supported_versions": [v.value for v in self.supported_versions],
            "default_version": self.default_version.value,
            "deprecated_versions": [v.value for v in self.deprecated_versions],
        }

    def add_versioned_router(
        self,
        app: FastAPI,
        router: APIRouter,
        version: APIVersion,
        prefix: str = "",
        tags: list[str] = None,
    ) -> None:
        """Add a router with version prefix."""
        if not self.is_supported(version):
            raise ValueError(f"Unsupported API version: {version.value}")

        version_prefix = self.get_version_prefix(version)
        full_prefix = f"{version_prefix}{prefix}" if prefix else version_prefix

        app.include_router(router, prefix=full_prefix, tags=tags or [])


# Global version manager instance
version_manager = VersionManager()


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    return version_manager
