"""Persistence layer implementations."""

from .connection_pool_integration import DatabaseConnectionIntegration
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
from .production_database import (
    ProductionDatabaseManager,
    get_production_database_manager,
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
    # Production database
    "ProductionDatabaseManager",
    "get_production_database_manager",
    "DatabaseConnectionIntegration",
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
