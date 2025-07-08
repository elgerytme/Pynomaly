"""Repository implementations."""

from .async_wrappers import (
    AsyncDatasetRepositoryWrapper,
    AsyncDetectionResultRepositoryWrapper,
    AsyncDetectorRepositoryWrapper,
)
from .dataset_repository import DatasetRepository
from .detector_repository import DetectorRepository
from .file_repositories import (
    FileDatasetRepository,
    FileDetectorRepository,
    FileResultRepository,
)
from .in_memory_repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)
from .memory_repository import (
    FileSystemDetectorRepository,
    MemoryDatasetRepository,
    MemoryDetectionResultRepository,
    MemoryDetectorRepository,
)
from .model_performance_repository import (
    ModelPerformanceRepository,
    PerformanceBaselineRepository,
)
from .repository_factory import RepositoryFactory
from .repository_service import (
    RepositoryService,
    create_filesystem_repository_service,
    create_memory_repository_service,
)

__all__ = [
    # Legacy implementations
    "InMemoryDetectorRepository",
    "InMemoryDatasetRepository",
    "InMemoryResultRepository",
    "FileDetectorRepository",
    "FileDatasetRepository",
    "FileResultRepository",
    "DetectorRepository",
    "DatasetRepository",
    "AsyncDetectorRepositoryWrapper",
    "AsyncDatasetRepositoryWrapper",
    "AsyncDetectionResultRepositoryWrapper",
    # New clean architecture implementations
    "MemoryDetectorRepository",
    "MemoryDatasetRepository",
    "MemoryDetectionResultRepository",
    "FileSystemDetectorRepository",
    # Service and factory
    "RepositoryService",
    "RepositoryFactory",
    # Performance repositories
    "ModelPerformanceRepository",
    "PerformanceBaselineRepository",
    # Convenience functions
    "create_memory_repository_service",
    "create_filesystem_repository_service",
]
