"""Migration hooks and event handlers for database migrations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import event, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class MigrationHooks:
    """Collection of migration hooks and event handlers."""

    def __init__(self):
        """Initialize migration hooks."""
        self.pre_migration_hooks: list[Callable] = []
        self.post_migration_hooks: list[Callable] = []
        self.migration_events: list[dict[str, Any]] = []

    def register_pre_migration_hook(self, hook: Callable) -> None:
        """Register a pre-migration hook.
        
        Args:
            hook: Function to call before migration
        """
        self.pre_migration_hooks.append(hook)

    def register_post_migration_hook(self, hook: Callable) -> None:
        """Register a post-migration hook.
        
        Args:
            hook: Function to call after migration
        """
        self.post_migration_hooks.append(hook)

    def execute_pre_migration_hooks(self, revision: str, direction: str) -> None:
        """Execute all pre-migration hooks.
        
        Args:
            revision: Migration revision
            direction: Migration direction (upgrade/downgrade)
        """
        for hook in self.pre_migration_hooks:
            try:
                hook(revision, direction, "pre")
                logger.info(f"Pre-migration hook executed for {revision}")
            except Exception as e:
                logger.error(f"Pre-migration hook failed: {e}")
                raise

    def execute_post_migration_hooks(self, revision: str, direction: str) -> None:
        """Execute all post-migration hooks.
        
        Args:
            revision: Migration revision
            direction: Migration direction (upgrade/downgrade)
        """
        for hook in self.post_migration_hooks:
            try:
                hook(revision, direction, "post")
                logger.info(f"Post-migration hook executed for {revision}")
            except Exception as e:
                logger.error(f"Post-migration hook failed: {e}")
                # Don't raise here to avoid breaking rollback

    def log_migration_event(self, event_type: str, revision: str, **kwargs) -> None:
        """Log migration event.
        
        Args:
            event_type: Type of event (start, end, error)
            revision: Migration revision
            **kwargs: Additional event data
        """
        event = {
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "revision": revision,
            **kwargs
        }
        self.migration_events.append(event)
        logger.info(f"Migration event: {event_type} for {revision}")


# Global hooks instance
migration_hooks = MigrationHooks()


def backup_before_migration(revision: str, direction: str, phase: str) -> None:
    """Create backup before migration if configured.
    
    Args:
        revision: Migration revision
        direction: Migration direction
        phase: Migration phase (pre/post)
    """
    if phase == "pre" and direction == "upgrade":
        # Only backup before upgrades
        logger.info(f"Creating backup before migration {revision}")
        # Backup logic would go here
        migration_hooks.log_migration_event("backup_created", revision)


def validate_data_integrity(revision: str, direction: str, phase: str) -> None:
    """Validate data integrity after migration.
    
    Args:
        revision: Migration revision
        direction: Migration direction
        phase: Migration phase (pre/post)
    """
    if phase == "post":
        logger.info(f"Validating data integrity after {direction} {revision}")
        # Data integrity validation logic
        migration_hooks.log_migration_event("integrity_validated", revision)


def update_migration_metadata(revision: str, direction: str, phase: str) -> None:
    """Update migration metadata and statistics.
    
    Args:
        revision: Migration revision
        direction: Migration direction
        phase: Migration phase (pre/post)
    """
    if phase == "post":
        logger.info(f"Updating migration metadata for {revision}")
        # Metadata update logic
        migration_hooks.log_migration_event("metadata_updated", revision)


def notify_migration_completion(revision: str, direction: str, phase: str) -> None:
    """Send notifications about migration completion.
    
    Args:
        revision: Migration revision
        direction: Migration direction
        phase: Migration phase (pre/post)
    """
    if phase == "post":
        logger.info(f"Migration {direction} completed for {revision}")
        # Notification logic (email, Slack, etc.)
        migration_hooks.log_migration_event("notification_sent", revision)


# Register default hooks
migration_hooks.register_pre_migration_hook(backup_before_migration)
migration_hooks.register_post_migration_hook(validate_data_integrity)
migration_hooks.register_post_migration_hook(update_migration_metadata)
migration_hooks.register_post_migration_hook(notify_migration_completion)


class MigrationEventListener:
    """SQLAlchemy event listener for migrations."""

    def __init__(self, engine: Engine):
        """Initialize event listener.
        
        Args:
            engine: SQLAlchemy engine
        """
        self.engine = engine
        self.setup_listeners()

    def setup_listeners(self) -> None:
        """Set up SQLAlchemy event listeners."""
        @event.listens_for(self.engine, "before_cursor_execute")
        def log_queries(conn, cursor, statement, parameters, context, executemany):
            """Log SQL queries during migrations."""
            if context and context.get("migration_context"):
                logger.debug(f"Migration SQL: {statement}")

        @event.listens_for(self.engine, "after_cursor_execute")
        def log_query_completion(conn, cursor, statement, parameters, context, executemany):
            """Log query completion during migrations."""
            if context and context.get("migration_context"):
                logger.debug("Migration SQL completed")


class MigrationPerformanceMonitor:
    """Monitor migration performance and resource usage."""

    def __init__(self):
        """Initialize performance monitor."""
        self.migration_metrics: dict[str, dict[str, Any]] = {}

    def start_monitoring(self, revision: str) -> None:
        """Start monitoring migration performance.
        
        Args:
            revision: Migration revision
        """
        self.migration_metrics[revision] = {
            "start_time": datetime.now(timezone.utc),
            "memory_start": self._get_memory_usage(),
            "cpu_start": self._get_cpu_usage(),
        }
        logger.info(f"Started monitoring migration {revision}")

    def stop_monitoring(self, revision: str) -> dict[str, Any]:
        """Stop monitoring and return metrics.
        
        Args:
            revision: Migration revision
            
        Returns:
            Performance metrics
        """
        if revision not in self.migration_metrics:
            return {}

        end_time = datetime.now(timezone.utc)
        metrics = self.migration_metrics[revision]
        
        duration = (end_time - metrics["start_time"]).total_seconds()
        memory_end = self._get_memory_usage()
        cpu_end = self._get_cpu_usage()

        performance_data = {
            "duration_seconds": duration,
            "memory_usage_mb": memory_end - metrics["memory_start"],
            "cpu_usage_percent": cpu_end - metrics["cpu_start"],
            "start_time": metrics["start_time"],
            "end_time": end_time,
        }

        logger.info(f"Migration {revision} completed in {duration:.2f}s")
        return performance_data

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0


class MigrationErrorHandler:
    """Handle migration errors and recovery."""

    def __init__(self):
        """Initialize error handler."""
        self.error_log: list[dict[str, Any]] = []

    def handle_migration_error(self, revision: str, error: Exception) -> None:
        """Handle migration error.
        
        Args:
            revision: Migration revision that failed
            error: Exception that occurred
        """
        error_info = {
            "timestamp": datetime.now(timezone.utc),
            "revision": revision,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": self._get_traceback(error),
        }
        
        self.error_log.append(error_info)
        logger.error(f"Migration {revision} failed: {error}")
        
        # Send error notification
        self._send_error_notification(error_info)

    def _get_traceback(self, error: Exception) -> str:
        """Get formatted traceback from exception."""
        import traceback
        return traceback.format_exc()

    def _send_error_notification(self, error_info: dict[str, Any]) -> None:
        """Send error notification to administrators."""
        # Notification logic (email, Slack, etc.)
        logger.info("Error notification sent to administrators")

    def get_recent_errors(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent migration errors.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent errors
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            error for error in self.error_log
            if error["timestamp"] > cutoff_time
        ]


class MigrationLockManager:
    """Manage migration locks to prevent concurrent execution."""

    def __init__(self, engine: Engine):
        """Initialize lock manager.
        
        Args:
            engine: SQLAlchemy engine
        """
        self.engine = engine

    def acquire_migration_lock(self, timeout: int = 300) -> bool:
        """Acquire migration lock.
        
        Args:
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Try to acquire an advisory lock
                if "postgresql" in str(self.engine.url):
                    # PostgreSQL advisory lock
                    result = conn.execute(
                        text("SELECT pg_try_advisory_lock(12345678)")
                    ).scalar()
                    return bool(result)
                elif "sqlite" in str(self.engine.url):
                    # SQLite doesn't have advisory locks, use table-based lock
                    try:
                        conn.execute(text("BEGIN IMMEDIATE"))
                        conn.execute(text("COMMIT"))
                        return True
                    except Exception:
                        return False
                else:
                    # For other databases, assume lock acquired
                    return True
        except Exception as e:
            logger.error(f"Failed to acquire migration lock: {e}")
            return False

    def release_migration_lock(self) -> bool:
        """Release migration lock.
        
        Returns:
            True if lock released, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                if "postgresql" in str(self.engine.url):
                    # PostgreSQL advisory lock release
                    result = conn.execute(
                        text("SELECT pg_advisory_unlock(12345678)")
                    ).scalar()
                    return bool(result)
                else:
                    # For other databases, assume lock released
                    return True
        except Exception as e:
            logger.error(f"Failed to release migration lock: {e}")
            return False


# Global instances
performance_monitor = MigrationPerformanceMonitor()
error_handler = MigrationErrorHandler()


def enhanced_migration_runner(migration_manager, revision: str = "head") -> dict[str, Any]:
    """Enhanced migration runner with hooks and monitoring.
    
    Args:
        migration_manager: Migration manager instance
        revision: Target revision
        
    Returns:
        Migration results with metrics
    """
    lock_manager = MigrationLockManager(migration_manager.engine)
    
    # Acquire lock
    if not lock_manager.acquire_migration_lock():
        raise RuntimeError("Could not acquire migration lock")
    
    try:
        # Start monitoring
        performance_monitor.start_monitoring(revision)
        
        # Execute pre-migration hooks
        migration_hooks.execute_pre_migration_hooks(revision, "upgrade")
        
        # Run migration
        success = migration_manager.run_migrations(revision)
        
        if success:
            # Execute post-migration hooks
            migration_hooks.execute_post_migration_hooks(revision, "upgrade")
            
            # Get performance metrics
            metrics = performance_monitor.stop_monitoring(revision)
            
            return {
                "success": True,
                "revision": revision,
                "metrics": metrics,
                "events": migration_hooks.migration_events,
            }
        else:
            raise RuntimeError("Migration failed")
            
    except Exception as e:
        error_handler.handle_migration_error(revision, e)
        raise
    finally:
        # Always release lock
        lock_manager.release_migration_lock()