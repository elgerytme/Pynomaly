"""Persistence layer implementations."""

from .database import (
    POSTGRESQL_LOCAL_URL,
    SQLITE_FILE_URL,
    SQLITE_MEMORY_URL,
    DatabaseManager,
    get_database_manager,
    get_session,
    init_database,
)
from .database_repositories import (
    Base,
    DatabaseDatasetRepository,
    DatabaseDetectionResultRepository,
    DatabaseDetectorRepository,
    DatasetModel,
    DetectionResultModel,
    DetectorModel,
)

__all__ = [
    # Database management
    "DatabaseManager",
    "init_database",
    "get_database_manager",
    "get_session",
    "SQLITE_MEMORY_URL",
    "SQLITE_FILE_URL",
    "POSTGRESQL_LOCAL_URL",
    # Repositories
    "DatabaseDetectorRepository",
    "DatabaseDatasetRepository",
    "DatabaseDetectionResultRepository",
    # Models
    "Base",
    "DatasetModel",
    "DetectorModel",
    "DetectionResultModel",
]
