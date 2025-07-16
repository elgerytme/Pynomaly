"""
Fast CLI Container Implementation

Lightweight container optimized for CLI startup performance.
Avoids heavy imports and database connections.
"""

from __future__ import annotations

import functools
from typing import Any

from pynomaly_detection.infrastructure.config.settings import Settings, get_settings


class FastCLIContainer:
    """Lightweight container for CLI operations."""

    def __init__(self):
        self._settings: Settings | None = None
        self._repositories: dict[str, Any] = {}
        self._services: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}

    @functools.lru_cache(maxsize=1)
    def config(self) -> Settings:
        """Get configuration settings (cached)."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def detector_repository(self):
        """Get detector repository (in-memory for CLI)."""
        if "detector" not in self._repositories:
            # Use in-memory repository for CLI to avoid database overhead
            from pynomaly_detection.infrastructure.persistence.memory_repository import (
                MemoryRepository,
            )

            self._repositories["detector"] = MemoryRepository()
        return self._repositories["detector"]

    def dataset_repository(self):
        """Get dataset repository (in-memory for CLI)."""
        if "dataset" not in self._repositories:
            from pynomaly_detection.infrastructure.persistence.memory_repository import (
                MemoryRepository,
            )

            self._repositories["dataset"] = MemoryRepository()
        return self._repositories["dataset"]

    def result_repository(self):
        """Get result repository (in-memory for CLI)."""
        if "result" not in self._repositories:
            from pynomaly_detection.infrastructure.persistence.memory_repository import (
                MemoryRepository,
            )

            self._repositories["result"] = MemoryRepository()
        return self._repositories["result"]

    def get_service(self, service_name: str, lazy: bool = True):
        """Get service with lazy loading."""
        if service_name in self._services:
            return self._services[service_name]

        if not lazy:
            # Only create service if explicitly requested
            return self._create_service(service_name)

        # Return a placeholder that will be resolved later
        return None

    def _create_service(self, service_name: str):
        """Create service instance on demand."""
        service_mapping = {
            "detector_service": self._create_detector_service,
            "dataset_service": self._create_dataset_service,
            "detection_service": self._create_detection_service,
            "training_service": self._create_training_service,
        }

        if service_name in service_mapping:
            service = service_mapping[service_name]()
            self._services[service_name] = service
            return service

        raise ValueError(f"Unknown service: {service_name}")

    def _create_detector_service(self):
        """Create detector service."""
        from pynomaly_detection.application.services.detector_service import DetectorService

        return DetectorService(self.detector_repository())

    def _create_dataset_service(self):
        """Create dataset service."""
        from pynomaly_detection.application.services.dataset_service import DatasetService

        return DatasetService(self.dataset_repository())

    def _create_detection_service(self):
        """Create detection service."""
        from pynomaly_detection.application.services.detection_service import DetectionService

        return DetectionService(
            self.detector_repository(),
            self.dataset_repository(),
            self.result_repository(),
        )

    def _create_training_service(self):
        """Create training service."""
        from pynomaly_detection.application.services.training_service import TrainingService

        return TrainingService(
            self.detector_repository(),
            self.dataset_repository(),
        )


# Global container instance for CLI (cached)
_cli_container: FastCLIContainer | None = None


def get_fast_cli_container() -> FastCLIContainer:
    """Get or create fast CLI container."""
    global _cli_container
    if _cli_container is None:
        _cli_container = FastCLIContainer()
    return _cli_container


def clear_cli_container_cache() -> None:
    """Clear CLI container cache (for testing)."""
    global _cli_container
    _cli_container = None
