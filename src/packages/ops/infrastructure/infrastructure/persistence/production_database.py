"""Production-ready database configuration with comprehensive connection pooling."""

from __future__ import annotations

import os
from typing import Any

import structlog
from sqlalchemy import event

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.performance.connection_pooling import (
    PoolConfiguration,
    get_connection_pool_manager,
)

from .enhanced_database import DatabaseHealthMonitor, EnhancedDatabaseManager

logger = structlog.get_logger(__name__)


class ProductionPoolConfiguration(PoolConfiguration):
    """Production-optimized pool configuration."""

    def __init__(self, environment: str = "production"):
        """Initialize production pool configuration.

        Args:
            environment: Deployment environment (development, testing, production)
        """
        if environment == "production":
            super().__init__(
                min_size=20,  # Higher minimum connections for production
                max_size=100,  # Higher maximum connections
                timeout=10.0,  # Faster timeout for production
                max_overflow=50,  # Allow more overflow connections
                recycle_time=1800,  # Recycle connections every 30 minutes
                pre_ping=True,  # Always verify connections
                pool_reset_on_return="commit",
                pool_timeout=10.0,
                max_idle_time=300,  # 5 minutes idle time
                health_check_interval=30,  # Check health every 30 seconds
                max_retries=5,  # More retries for production
                retry_delay=0.5,  # Faster retry delay
                backoff_factor=1.5,  # Moderate backoff
                enable_metrics=True,
                log_slow_queries=True,
                slow_query_threshold=0.5,  # Log queries slower than 500ms
            )
        elif environment == "testing":
            super().__init__(
                min_size=2,
                max_size=5,
                timeout=5.0,
                max_overflow=2,
                recycle_time=300,
                pre_ping=False,
                health_check_interval=0,  # Disable health checks in testing
                enable_metrics=False,
                log_slow_queries=False,
            )
        else:  # development
            super().__init__(
                min_size=5,
                max_size=20,
                timeout=30.0,
                max_overflow=10,
                recycle_time=3600,
                pre_ping=True,
                health_check_interval=60,
                enable_metrics=True,
                log_slow_queries=True,
                slow_query_threshold=1.0,
            )


class ProductionDatabaseManager:
    """Production-ready database manager with comprehensive features."""

    def __init__(
        self,
        settings: Settings | None = None,
        environment: str | None = None,
        enable_monitoring: bool = True,
    ):
        """Initialize production database manager.

        Args:
            settings: Application settings
            environment: Deployment environment
            enable_monitoring: Whether to enable health monitoring
        """
        self.settings = settings or Settings()
        self.environment = environment or os.getenv("PYNOMALY_ENV", "development")
        self.enable_monitoring = enable_monitoring

        # Connection pool configuration
        self.pool_config = ProductionPoolConfiguration(self.environment)

        # Database URL
        self.database_url = self._get_database_url()

        # Enhanced database manager
        self.db_manager = EnhancedDatabaseManager(
            database_url=self.database_url,
            pool_config=self.pool_config,
            enable_query_optimization=True,
        )

        # Health monitoring
        self.health_monitor = None
        if enable_monitoring:
            self.health_monitor = DatabaseHealthMonitor(
                db_manager=self.db_manager,
                check_interval=self.pool_config.health_check_interval,
            )
            self.health_monitor.start_monitoring()

        # Connection pool manager
        self.pool_manager = get_connection_pool_manager()

        # Setup database event listeners
        self._setup_database_listeners()

        logger.info(
            "Production database manager initialized",
            environment=self.environment,
            pool_min_size=self.pool_config.min_size,
            pool_max_size=self.pool_config.max_size,
            health_monitoring=enable_monitoring,
        )

    def _get_database_url(self) -> str:
        """Get database URL based on environment."""
        # Check environment variable first
        database_url = os.getenv("PYNOMALY_DATABASE_URL")
        if database_url:
            return database_url

        # Use settings
        if hasattr(self.settings, "database_url") and self.settings.database_url:
            return self.settings.database_url

        # Environment-specific defaults
        if self.environment == "production":
            # Production should always have an explicit URL
            raise ValueError(
                "Production database URL not configured. "
                "Set PYNOMALY_DATABASE_URL environment variable."
            )
        elif self.environment == "testing":
            return "sqlite+aiosqlite:///:memory:"
        else:  # development
            return "sqlite+aiosqlite:///./pynomaly_dev.db"

    def _setup_database_listeners(self) -> None:
        """Setup database event listeners for monitoring and optimization."""
        if not self.db_manager._database_pool:
            return

        engine = self.db_manager._database_pool._sync_engine
        if not engine:
            return

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for performance and reliability."""
            if self.database_url.startswith("sqlite:"):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()

        @event.listens_for(engine, "connect")
        def set_postgresql_settings(dbapi_connection, connection_record):
            """Set PostgreSQL settings for performance."""
            if self.database_url.startswith("postgresql"):
                cursor = dbapi_connection.cursor()
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("SET lock_timeout = '10s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")
                cursor.close()

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout."""
            if self.pool_config.enable_metrics:
                logger.debug("Database connection checked out")

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Handle connection checkin."""
            if self.pool_config.enable_metrics:
                logger.debug("Database connection checked in")

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {
            "environment": self.environment,
            "database_url": self.database_url.split("@")[-1]
            if "@" in self.database_url
            else self.database_url,
            "pool_config": {
                "min_size": self.pool_config.min_size,
                "max_size": self.pool_config.max_size,
                "timeout": self.pool_config.timeout,
                "max_overflow": self.pool_config.max_overflow,
                "recycle_time": self.pool_config.recycle_time,
                "health_check_interval": self.pool_config.health_check_interval,
            },
            "pool_stats": self.db_manager.get_pool_stats(),
            "query_stats": self.db_manager.get_query_stats(),
        }

        # Add health monitoring stats
        if self.health_monitor:
            stats["health_monitoring"] = {
                "enabled": True,
                "summary": self.health_monitor.get_health_summary(),
                "recent_history": self.health_monitor.get_health_history(limit=5),
            }
        else:
            stats["health_monitoring"] = {"enabled": False}

        return stats

    async def perform_maintenance(self) -> dict[str, Any]:
        """Perform database maintenance tasks."""
        maintenance_results = {
            "timestamp": __import__("time").time(),
            "tasks_completed": [],
            "errors": [],
        }

        try:
            # Clear query cache
            await self.db_manager.clear_query_cache()
            maintenance_results["tasks_completed"].append("query_cache_cleared")

            # Get optimization report
            optimization_report = await self.db_manager.get_optimization_report()
            maintenance_results["optimization_report"] = optimization_report
            maintenance_results["tasks_completed"].append(
                "optimization_report_generated"
            )

            # Perform database optimization
            optimization_results = await self.db_manager.optimize_database()
            maintenance_results["optimization_results"] = optimization_results
            maintenance_results["tasks_completed"].append("database_optimized")

            # Reset pool statistics
            self.pool_manager.reset_stats()
            maintenance_results["tasks_completed"].append("pool_stats_reset")

            logger.info("Database maintenance completed", results=maintenance_results)

        except Exception as e:
            error_msg = f"Database maintenance error: {str(e)}"
            maintenance_results["errors"].append(error_msg)
            logger.error("Database maintenance failed", error=str(e))

        return maintenance_results

    async def emergency_pool_reset(self) -> dict[str, Any]:
        """Emergency reset of connection pools."""
        reset_results = {
            "timestamp": __import__("time").time(),
            "actions_taken": [],
            "status": "success",
        }

        try:
            # Close current database pool
            await self.pool_manager.close_pool("main_database")
            reset_results["actions_taken"].append("closed_existing_pool")

            # Recreate database pool
            self.db_manager._database_pool = self.pool_manager.create_database_pool(
                name="main_database",
                database_url=self.database_url,
                config=self.pool_config,
                async_mode=True,
            )
            reset_results["actions_taken"].append("recreated_pool")

            # Test new pool
            health_status = await self.db_manager.health_check()
            if health_status["database_available"]:
                reset_results["actions_taken"].append("pool_health_verified")
            else:
                reset_results["status"] = "failed"
                reset_results["error"] = "New pool failed health check"

            logger.info("Emergency pool reset completed", results=reset_results)

        except Exception as e:
            reset_results["status"] = "failed"
            reset_results["error"] = str(e)
            logger.error("Emergency pool reset failed", error=str(e))

        return reset_results

    async def scale_pool(self, new_min_size: int, new_max_size: int) -> dict[str, Any]:
        """Dynamically scale connection pool size."""
        scale_results = {
            "timestamp": __import__("time").time(),
            "old_config": {
                "min_size": self.pool_config.min_size,
                "max_size": self.pool_config.max_size,
            },
            "new_config": {
                "min_size": new_min_size,
                "max_size": new_max_size,
            },
            "status": "success",
        }

        try:
            # Update pool configuration
            self.pool_config.min_size = new_min_size
            self.pool_config.max_size = new_max_size

            # Recreate pool with new configuration
            await self.emergency_pool_reset()

            logger.info(
                "Database pool scaled",
                old_min=scale_results["old_config"]["min_size"],
                old_max=scale_results["old_config"]["max_size"],
                new_min=new_min_size,
                new_max=new_max_size,
            )

        except Exception as e:
            scale_results["status"] = "failed"
            scale_results["error"] = str(e)
            logger.error("Database pool scaling failed", error=str(e))

        return scale_results

    async def close(self) -> None:
        """Close database manager and cleanup resources."""
        try:
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()

            # Close database manager
            await self.db_manager.close()

            logger.info("Production database manager closed")

        except Exception as e:
            logger.error("Error closing production database manager", error=str(e))


# Global production database manager
_production_db_manager: ProductionDatabaseManager | None = None


def get_production_database_manager(
    settings: Settings | None = None,
    environment: str | None = None,
    enable_monitoring: bool = True,
) -> ProductionDatabaseManager:
    """Get or create global production database manager.

    Args:
        settings: Application settings
        environment: Deployment environment
        enable_monitoring: Whether to enable health monitoring

    Returns:
        Production database manager instance
    """
    global _production_db_manager

    if _production_db_manager is None:
        _production_db_manager = ProductionDatabaseManager(
            settings=settings,
            environment=environment,
            enable_monitoring=enable_monitoring,
        )

    return _production_db_manager


async def close_production_database_manager() -> None:
    """Close global production database manager."""
    global _production_db_manager

    if _production_db_manager:
        await _production_db_manager.close()
        _production_db_manager = None
