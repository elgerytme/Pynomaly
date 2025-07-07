"""Repository factory for creating repository instances."""

from __future__ import annotations

import os
from typing import Literal

from pynomaly.infrastructure.repositories.repository_service import RepositoryService


class RepositoryFactory:
    """Factory for creating repository implementations."""

    @staticmethod
    def create_repository_service(
        storage_type: Literal["memory", "filesystem"] = "memory",
        base_path: str = "data",
    ) -> RepositoryService:
        """Create repository service with specified storage backend.

        Args:
            storage_type: Type of storage backend ("memory" or "filesystem")
            base_path: Base path for file storage (only used for filesystem)

        Returns:
            Configured repository service
        """
        if storage_type == "memory":
            return _create_memory_repositories()
        elif storage_type == "filesystem":
            return _create_filesystem_repositories(base_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    @staticmethod
    def create_from_environment() -> RepositoryService:
        """Create repository service based on environment variables.

        Environment variables:
        - PYNOMALY_STORAGE_TYPE: "memory" or "filesystem" (default: "memory")
        - PYNOMALY_DATA_PATH: Base path for data storage (default: "data")

        Returns:
            Configured repository service
        """
        storage_type = os.getenv("PYNOMALY_STORAGE_TYPE", "memory")
        base_path = os.getenv("PYNOMALY_DATA_PATH", "data")

        if storage_type not in ["memory", "filesystem"]:
            storage_type = "memory"  # Fallback to safe default

        return RepositoryFactory.create_repository_service(
            storage_type=storage_type, base_path=base_path  # type: ignore
        )

    @staticmethod
    def create_test_repositories() -> RepositoryService:
        """Create repository service optimized for testing.

        Returns:
            In-memory repository service for testing
        """
        return _create_memory_repositories()


def _create_memory_repositories() -> RepositoryService:
    """Create repository service with in-memory implementations."""
    from pynomaly.infrastructure.repositories.memory_repository import (
        MemoryDatasetRepository,
        MemoryDetectionResultRepository,
        MemoryDetectorRepository,
    )

    return RepositoryService(
        detector_repository=MemoryDetectorRepository(),
        dataset_repository=MemoryDatasetRepository(),
        result_repository=MemoryDetectionResultRepository(),
    )


def _create_filesystem_repositories(base_path: str) -> RepositoryService:
    """Create repository service with file system implementations."""
    from pynomaly.infrastructure.repositories.memory_repository import (
        FileSystemDetectorRepository,
        MemoryDatasetRepository,
        MemoryDetectionResultRepository,
    )

    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)

    return RepositoryService(
        detector_repository=FileSystemDetectorRepository(f"{base_path}/detectors"),
        dataset_repository=MemoryDatasetRepository(f"{base_path}/datasets"),
        result_repository=MemoryDetectionResultRepository(),
    )
