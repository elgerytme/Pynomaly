"""
Model Repository Implementation

Persistent storage implementation for model versions and model metadata
with support for in-memory storage for testing and development.
"""

import logging
from abc import ABC, abstractmethod

from pynomaly.domain.entities.model_version import ModelVersion

logger = logging.getLogger(__name__)


class ModelRepositoryProtocol(ABC):
    """Protocol for model repositories."""

    @abstractmethod
    async def save_version(self, model_version: ModelVersion) -> None:
        """Save a model version."""
        pass

    @abstractmethod
    async def get_version(self, version_id: str) -> ModelVersion | None:
        """Get a model version by ID."""
        pass

    @abstractmethod
    async def list_versions(
        self, algorithm: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[ModelVersion]:
        """List model versions with optional filtering."""
        pass

    @abstractmethod
    async def delete_version(self, version_id: str) -> bool:
        """Delete a model version."""
        pass


class InMemoryModelRepository(ModelRepositoryProtocol):
    """In-memory implementation of model repository for testing and development."""

    def __init__(self):
        self._models: dict[str, ModelVersion] = {}

    async def save_version(self, model_version: ModelVersion) -> None:
        """Save a model version."""
        self._models[model_version.id] = model_version
        logger.debug(f"Saved model version {model_version.id}")

    async def get_version(self, version_id: str) -> ModelVersion | None:
        """Get a model version by ID."""
        return self._models.get(version_id)

    async def list_versions(
        self, algorithm: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[ModelVersion]:
        """List model versions with optional filtering."""
        versions = list(self._models.values())

        # Apply filters
        if algorithm:
            versions = [v for v in versions if v.algorithm == algorithm]

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        # Apply pagination
        return versions[offset : offset + limit]

    async def delete_version(self, version_id: str) -> bool:
        """Delete a model version."""
        if version_id in self._models:
            del self._models[version_id]
            logger.debug(f"Deleted model version {version_id}")
            return True
        return False


# Alias for backwards compatibility
ModelRepository = InMemoryModelRepository
