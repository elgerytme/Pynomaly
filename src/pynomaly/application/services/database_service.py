"""Database service for managing database connections and repositories."""

from __future__ import annotations

import logging
from typing import Any, Dict, Protocol

from pynomaly.domain.repositories.dataset_repository import DatasetRepository
from pynomaly.domain.repositories.detector_repository import DetectorRepository
from pynomaly.domain.repositories.detection_result_repository import DetectionResultRepository
from pynomaly.infrastructure.persistence.database_config import DatabaseSettings, get_database_settings
from pynomaly.infrastructure.persistence.database_factory import create_database_repository_service
from pynomaly.infrastructure.persistence.mongodb_repositories import MongoDBManager, create_mongodb_repository_service

logger = logging.getLogger(__name__)


class DatabaseService(Protocol):
    """Protocol for database service."""
    
    def get_detector_repository(self) -> DetectorRepository:
        """Get detector repository."""
        ...
    
    def get_dataset_repository(self) -> DatasetRepository:
        """Get dataset repository."""
        ...
    
    def get_detection_result_repository(self) -> DetectionResultRepository:
        """Get detection result repository."""
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        ...
    
    def close(self) -> None:
        """Close database connections."""
        ...


class PostgreSQLDatabaseService:
    """Database service for PostgreSQL."""
    
    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialize PostgreSQL database service."""
        self.settings = settings
        self.repository_service = create_database_repository_service(
            url=settings.postgresql.connection_url,
            **settings.postgresql.connection_params
        )
        logger.info("PostgreSQL database service initialized")
    
    def get_detector_repository(self) -> DetectorRepository:
        """Get detector repository."""
        return self.repository_service.detector_repository
    
    def get_dataset_repository(self) -> DatasetRepository:
        """Get dataset repository."""
        return self.repository_service.dataset_repository
    
    def get_detection_result_repository(self) -> DetectionResultRepository:
        """Get detection result repository."""
        return self.repository_service.detection_result_repository
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Test database connection
            detector_count = self.get_detector_repository().count()
            dataset_count = self.get_dataset_repository().count()
            result_count = self.get_detection_result_repository().count()
            
            return {
                "status": "healthy",
                "database_type": "postgresql",
                "connection_url": self.settings.postgresql.connection_url.replace(
                    self.settings.postgresql.password, "***"
                ),
                "detector_count": detector_count,
                "dataset_count": dataset_count,
                "result_count": result_count,
            }
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_type": "postgresql",
                "error": str(e),
            }
    
    def close(self) -> None:
        """Close database connections."""
        self.repository_service.close()
        logger.info("PostgreSQL database service closed")


class MongoDBDatabaseService:
    """Database service for MongoDB."""
    
    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialize MongoDB database service."""
        self.settings = settings
        self.mongodb_manager = MongoDBManager(
            connection_url=settings.mongodb.connection_url,
            **settings.mongodb.connection_params
        )
        self.repository_service = create_mongodb_repository_service(self.mongodb_manager)
        logger.info("MongoDB database service initialized")
    
    def get_detector_repository(self) -> DetectorRepository:
        """Get detector repository."""
        return self.repository_service.detector_repository
    
    def get_dataset_repository(self) -> DatasetRepository:
        """Get dataset repository."""
        return self.repository_service.dataset_repository
    
    def get_detection_result_repository(self) -> DetectionResultRepository:
        """Get detection result repository."""
        return self.repository_service.detection_result_repository
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Test database connection
            detector_count = self.get_detector_repository().count()
            dataset_count = self.get_dataset_repository().count()
            result_count = self.get_detection_result_repository().count()
            
            return {
                "status": "healthy",
                "database_type": "mongodb",
                "connection_url": self.settings.mongodb.connection_url.replace(
                    self.settings.mongodb.password or "", "***"
                ),
                "detector_count": detector_count,
                "dataset_count": dataset_count,
                "result_count": result_count,
            }
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_type": "mongodb",
                "error": str(e),
            }
    
    def close(self) -> None:
        """Close database connections."""
        self.mongodb_manager.close()
        logger.info("MongoDB database service closed")


class SQLiteDatabaseService:
    """Database service for SQLite."""
    
    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialize SQLite database service."""
        self.settings = settings
        self.repository_service = create_database_repository_service(
            url=f"sqlite:///{settings.sqlite_path}"
        )
        logger.info(f"SQLite database service initialized: {settings.sqlite_path}")
    
    def get_detector_repository(self) -> DetectorRepository:
        """Get detector repository."""
        return self.repository_service.detector_repository
    
    def get_dataset_repository(self) -> DatasetRepository:
        """Get dataset repository."""
        return self.repository_service.dataset_repository
    
    def get_detection_result_repository(self) -> DetectionResultRepository:
        """Get detection result repository."""
        return self.repository_service.detection_result_repository
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Test database connection
            detector_count = self.get_detector_repository().count()
            dataset_count = self.get_dataset_repository().count()
            result_count = self.get_detection_result_repository().count()
            
            return {
                "status": "healthy",
                "database_type": "sqlite",
                "database_path": self.settings.sqlite_path,
                "detector_count": detector_count,
                "dataset_count": dataset_count,
                "result_count": result_count,
            }
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_type": "sqlite",
                "error": str(e),
            }
    
    def close(self) -> None:
        """Close database connections."""
        self.repository_service.close()
        logger.info("SQLite database service closed")


def create_database_service(settings: DatabaseSettings | None = None) -> DatabaseService:
    """Create database service based on settings."""
    if settings is None:
        settings = get_database_settings()
    
    logger.info(f"Creating database service for: {settings.database_type}")
    
    if settings.database_type == "postgresql":
        return PostgreSQLDatabaseService(settings)
    elif settings.database_type == "mongodb":
        return MongoDBDatabaseService(settings)
    elif settings.database_type == "sqlite":
        return SQLiteDatabaseService(settings)
    else:
        raise ValueError(f"Unsupported database type: {settings.database_type}")


# Global database service instance
_database_service: DatabaseService | None = None


def get_database_service() -> DatabaseService:
    """Get or create the global database service instance."""
    global _database_service
    
    if _database_service is None:
        _database_service = create_database_service()
    
    return _database_service


def close_database_service() -> None:
    """Close the global database service instance."""
    global _database_service
    
    if _database_service is not None:
        _database_service.close()
        _database_service = None
        logger.info("Global database service closed")


# Database service factory for dependency injection
class DatabaseServiceFactory:
    """Factory for creating database services."""
    
    @staticmethod
    def create_postgresql_service(settings: DatabaseSettings) -> PostgreSQLDatabaseService:
        """Create PostgreSQL database service."""
        return PostgreSQLDatabaseService(settings)
    
    @staticmethod
    def create_mongodb_service(settings: DatabaseSettings) -> MongoDBDatabaseService:
        """Create MongoDB database service."""
        return MongoDBDatabaseService(settings)
    
    @staticmethod
    def create_sqlite_service(settings: DatabaseSettings) -> SQLiteDatabaseService:
        """Create SQLite database service."""
        return SQLiteDatabaseService(settings)
    
    @staticmethod
    def create_service(settings: DatabaseSettings) -> DatabaseService:
        """Create database service based on settings."""
        return create_database_service(settings)
