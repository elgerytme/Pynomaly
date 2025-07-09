"""Integration layer for database connection pools across the application."""

from __future__ import annotations

import os
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.performance.connection_pooling import (
    PoolConfiguration,
    get_connection_pool_manager,
)
from pynomaly.infrastructure.repositories.repository_factory import (
    RepositoryFactory,
)

from .enhanced_database import EnhancedDatabaseManager
from .production_database import (
    ProductionDatabaseManager,
    get_production_database_manager,
)

logger = structlog.get_logger(__name__)


class DatabaseConnectionIntegration:
    """Integration layer for database connections across the application."""

    def __init__(
        self,
        settings: Settings | None = None,
        environment: str | None = None,
        force_production_mode: bool = False,
    ):
        """Initialize database connection integration.

        Args:
            settings: Application settings
            environment: Deployment environment
            force_production_mode: Force production-level configuration
        """
        self.settings = settings or Settings()
        self.environment = environment or os.getenv("PYNOMALY_ENV", "development")
        self.force_production_mode = force_production_mode

        # Use production manager if in production or forced
        self.use_production_manager = (
            self.environment == "production" or force_production_mode
        )

        # Initialize managers
        self.production_manager: ProductionDatabaseManager | None = None
        self.enhanced_manager: EnhancedDatabaseManager | None = None
        self.pool_manager = get_connection_pool_manager()

        # Repository factory
        self.repository_factory: RepositoryFactory | None = None

        # Initialize the appropriate manager
        self._initialize_managers()

    def _initialize_managers(self) -> None:
        """Initialize database managers based on environment."""
        try:
            if self.use_production_manager:
                # Production manager with full monitoring
                self.production_manager = get_production_database_manager(
                    settings=self.settings,
                    environment=self.environment,
                    enable_monitoring=True,
                )

                # Repository factory using production manager
                self.repository_factory = RepositoryFactory()

                logger.info(
                    "Initialized production database manager",
                    environment=self.environment,
                    monitoring=True,
                )

            else:
                # Enhanced manager for development/testing
                database_url = self._get_database_url()
                pool_config = self._get_pool_config()

                self.enhanced_manager = EnhancedDatabaseManager(
                    database_url=database_url,
                    pool_config=pool_config,
                    enable_query_optimization=True,
                )

                # Repository factory using enhanced manager
                self.repository_factory = RepositoryFactory()

                logger.info(
                    "Initialized enhanced database manager",
                    environment=self.environment,
                    optimization=True,
                )

        except Exception as e:
            logger.error("Failed to initialize database managers", error=str(e))

            # Fallback to in-memory repositories
            self.repository_factory = RepositoryFactory()

            logger.warning(
                "Fallback to in-memory repositories",
                reason=str(e),
            )

    def _get_database_url(self) -> str:
        """Get database URL for non-production environments."""
        database_url = os.getenv("PYNOMALY_DATABASE_URL")
        if database_url:
            return database_url

        if self.environment == "testing":
            return "sqlite+aiosqlite:///:memory:"
        else:  # development
            return "sqlite+aiosqlite:///./pynomaly_dev.db"

    def _get_pool_config(self) -> PoolConfiguration:
        """Get pool configuration for non-production environments."""
        if self.environment == "testing":
            return PoolConfiguration(
                min_size=2,
                max_size=5,
                timeout=5.0,
                max_overflow=2,
                health_check_interval=0,
                enable_metrics=False,
            )
        else:  # development
            return PoolConfiguration(
                min_size=5,
                max_size=20,
                timeout=30.0,
                max_overflow=10,
                health_check_interval=60,
                enable_metrics=True,
            )

    @property
    def database_manager(self) -> EnhancedDatabaseManager:
        """Get the active database manager."""
        if self.production_manager:
            return self.production_manager.db_manager
        elif self.enhanced_manager:
            return self.enhanced_manager
        else:
            raise RuntimeError("No database manager initialized")

    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.database_manager.get_session() as session:
            return session

    async def get_repositories(self) -> dict[str, Any]:
        """Get all repositories."""
        if not self.repository_factory:
            raise RuntimeError("Repository factory not initialized")

        # Use the factory to create repository service
        repository_service = self.repository_factory.create_repository_service()

        return {
            "detector": repository_service.detector_repository,
            "dataset": repository_service.dataset_repository,
            "result": repository_service.result_repository,
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {
            "environment": self.environment,
            "production_mode": self.use_production_manager,
            "timestamp": __import__("time").time(),
        }

        try:
            # Database health
            db_health = await self.database_manager.health_check()
            health_status["database"] = db_health

            # Pool statistics
            pool_stats = self.database_manager.get_pool_stats()
            health_status["connection_pool"] = pool_stats

            # Query statistics
            query_stats = self.database_manager.get_query_stats()
            health_status["query_performance"] = query_stats

            # Production manager specific stats
            if self.production_manager:
                production_stats = self.production_manager.get_comprehensive_stats()
                health_status["production_stats"] = production_stats

            # Overall status
            health_status["overall_status"] = (
                "healthy" if db_health.get("database_available", False) else "unhealthy"
            )

        except Exception as e:
            health_status["error"] = str(e)
            health_status["overall_status"] = "error"
            logger.error("Failed to get health status", error=str(e))

        return health_status

    async def perform_maintenance(self) -> dict[str, Any]:
        """Perform database maintenance."""
        if self.production_manager:
            return await self.production_manager.perform_maintenance()
        elif self.enhanced_manager:
            # Basic maintenance for enhanced manager
            maintenance_results = {
                "timestamp": __import__("time").time(),
                "tasks_completed": [],
            }

            try:
                await self.enhanced_manager.clear_query_cache()
                maintenance_results["tasks_completed"].append("query_cache_cleared")

                optimization_report = (
                    await self.enhanced_manager.get_optimization_report()
                )
                maintenance_results["optimization_report"] = optimization_report
                maintenance_results["tasks_completed"].append(
                    "optimization_report_generated"
                )

            except Exception as e:
                maintenance_results["error"] = str(e)
                logger.error("Enhanced manager maintenance failed", error=str(e))

            return maintenance_results
        else:
            return {"error": "No database manager available for maintenance"}

    async def emergency_recovery(self) -> dict[str, Any]:
        """Perform emergency database recovery."""
        if self.production_manager:
            return await self.production_manager.emergency_pool_reset()
        else:
            # For enhanced manager, try to recreate connection pool
            try:
                await self.pool_manager.close_pool("main_database")

                # Reinitialize managers
                self._initialize_managers()

                return {
                    "timestamp": __import__("time").time(),
                    "status": "success",
                    "actions_taken": ["reinitialized_managers"],
                }
            except Exception as e:
                return {
                    "timestamp": __import__("time").time(),
                    "status": "failed",
                    "error": str(e),
                }

    async def scale_connections(self, min_size: int, max_size: int) -> dict[str, Any]:
        """Scale database connections."""
        if self.production_manager:
            return await self.production_manager.scale_pool(min_size, max_size)
        else:
            return {"error": "Connection scaling only available in production mode"}

    async def close(self) -> None:
        """Close all database connections."""
        try:
            if self.production_manager:
                await self.production_manager.close()
            elif self.enhanced_manager:
                await self.enhanced_manager.close()

            # Close all pools
            await self.pool_manager.close_all_pools()

            logger.info("Database connection integration closed")

        except Exception as e:
            logger.error("Error closing database connection integration", error=str(e))


# Global database integration instance
_database_integration: DatabaseConnectionIntegration | None = None


def get_database_integration(
    settings: Settings | None = None,
    environment: str | None = None,
    force_production_mode: bool = False,
) -> DatabaseConnectionIntegration:
    """Get or create global database connection integration.

    Args:
        settings: Application settings
        environment: Deployment environment
        force_production_mode: Force production-level configuration

    Returns:
        Database connection integration instance
    """
    global _database_integration

    if _database_integration is None:
        _database_integration = DatabaseConnectionIntegration(
            settings=settings,
            environment=environment,
            force_production_mode=force_production_mode,
        )

    return _database_integration


async def close_database_integration() -> None:
    """Close global database connection integration."""
    global _database_integration

    if _database_integration:
        await _database_integration.close()
        _database_integration = None


# Convenience functions for common operations
async def get_db_session() -> AsyncSession:
    """Get database session from global integration."""
    integration = get_database_integration()
    return await integration.get_session()


async def get_db_repositories() -> dict[str, Any]:
    """Get repositories from global integration."""
    integration = get_database_integration()
    return await integration.get_repositories()


async def get_db_health() -> dict[str, Any]:
    """Get database health status from global integration."""
    integration = get_database_integration()
    return await integration.get_health_status()


async def perform_db_maintenance() -> dict[str, Any]:
    """Perform database maintenance from global integration."""
    integration = get_database_integration()
    return await integration.perform_maintenance()


async def emergency_db_recovery() -> dict[str, Any]:
    """Perform emergency database recovery from global integration."""
    integration = get_database_integration()
    return await integration.emergency_recovery()
