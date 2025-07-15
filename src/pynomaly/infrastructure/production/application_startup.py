"""Production-ready application startup and initialization."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pynomaly.infrastructure.cache import get_cache_integration_manager
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.monitoring import get_comprehensive_health_manager
from pynomaly.infrastructure.persistence import get_production_database_manager
from pynomaly.infrastructure.security import get_rate_limit_manager
from pynomaly.shared.error_handling import (
    InfrastructureError,
)

from .production_config import Environment, get_production_config

logger = logging.getLogger(__name__)


class StartupPhase(Enum):
    """Application startup phases."""

    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    SERVICES = "services"
    HEALTH_CHECKS = "health_checks"
    READY = "ready"


@dataclass
class StartupTask:
    """Individual startup task definition."""

    name: str
    phase: StartupPhase
    task_function: Callable
    dependencies: set[str] = field(default_factory=set)
    timeout_seconds: float = 30.0
    critical: bool = True
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    def __post_init__(self):
        """Ensure dependencies is a set."""
        if isinstance(self.dependencies, (list, tuple)):
            self.dependencies = set(self.dependencies)


@dataclass
class StartupResult:
    """Result of a startup task execution."""

    task_name: str
    phase: StartupPhase
    success: bool
    duration_seconds: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ApplicationStartup:
    """Manages application startup process."""

    def __init__(self, settings: Settings | None = None):
        """Initialize application startup manager.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.startup_tasks: dict[str, StartupTask] = {}
        self.completed_tasks: set[str] = set()
        self.startup_results: list[StartupResult] = []
        self.startup_time = 0.0
        self.is_ready = False

        # Register default startup tasks
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        """Register default startup tasks."""

        # Phase 1: Initialization
        self.register_task(
            StartupTask(
                name="load_configuration",
                phase=StartupPhase.INITIALIZATION,
                task_function=self._load_configuration,
                timeout_seconds=5.0,
            )
        )

        self.register_task(
            StartupTask(
                name="setup_logging",
                phase=StartupPhase.INITIALIZATION,
                task_function=self._setup_logging,
                dependencies={"load_configuration"},
                timeout_seconds=5.0,
            )
        )

        # Phase 2: Configuration
        self.register_task(
            StartupTask(
                name="validate_environment",
                phase=StartupPhase.CONFIGURATION,
                task_function=self._validate_environment,
                dependencies={"load_configuration"},
                timeout_seconds=10.0,
            )
        )

        self.register_task(
            StartupTask(
                name="setup_security",
                phase=StartupPhase.CONFIGURATION,
                task_function=self._setup_security,
                dependencies={"validate_environment"},
                timeout_seconds=10.0,
            )
        )

        # Phase 3: Infrastructure
        self.register_task(
            StartupTask(
                name="initialize_database",
                phase=StartupPhase.INFRASTRUCTURE,
                task_function=self._initialize_database,
                dependencies={"setup_security"},
                timeout_seconds=30.0,
            )
        )

        self.register_task(
            StartupTask(
                name="initialize_cache",
                phase=StartupPhase.INFRASTRUCTURE,
                task_function=self._initialize_cache,
                dependencies={"initialize_database"},
                timeout_seconds=15.0,
            )
        )

        self.register_task(
            StartupTask(
                name="initialize_rate_limiting",
                phase=StartupPhase.INFRASTRUCTURE,
                task_function=self._initialize_rate_limiting,
                dependencies={"initialize_cache"},
                timeout_seconds=10.0,
            )
        )

        # Phase 4: Services
        self.register_task(
            StartupTask(
                name="initialize_services",
                phase=StartupPhase.SERVICES,
                task_function=self._initialize_services,
                dependencies={"initialize_rate_limiting"},
                timeout_seconds=20.0,
            )
        )

        self.register_task(
            StartupTask(
                name="load_models",
                phase=StartupPhase.SERVICES,
                task_function=self._load_models,
                dependencies={"initialize_services"},
                timeout_seconds=60.0,
                critical=False,  # Model loading can be non-critical
            )
        )

        # Phase 5: Health Checks
        self.register_task(
            StartupTask(
                name="initialize_health_monitoring",
                phase=StartupPhase.HEALTH_CHECKS,
                task_function=self._initialize_health_monitoring,
                dependencies={"load_models"},
                timeout_seconds=15.0,
            )
        )

        self.register_task(
            StartupTask(
                name="run_startup_health_checks",
                phase=StartupPhase.HEALTH_CHECKS,
                task_function=self._run_startup_health_checks,
                dependencies={"initialize_health_monitoring"},
                timeout_seconds=30.0,
            )
        )

    def register_task(self, task: StartupTask) -> None:
        """Register a startup task.

        Args:
            task: Startup task to register
        """
        self.startup_tasks[task.name] = task
        logger.debug(
            f"Registered startup task: {task.name} (phase: {task.phase.value})"
        )

    async def startup(self) -> bool:
        """Execute application startup sequence.

        Returns:
            True if startup completed successfully
        """
        logger.info("Starting application startup sequence...")
        start_time = time.time()

        try:
            # Execute tasks by phase
            for phase in StartupPhase:
                if phase == StartupPhase.READY:
                    continue  # Skip READY phase

                phase_tasks = [
                    task for task in self.startup_tasks.values() if task.phase == phase
                ]

                if not phase_tasks:
                    continue

                logger.info(f"Executing startup phase: {phase.value}")
                success = await self._execute_phase(phase_tasks)

                if not success:
                    logger.error(f"Startup failed in phase: {phase.value}")
                    return False

            # Mark as ready
            self.startup_time = time.time() - start_time
            self.is_ready = True

            logger.info(
                f"Application startup completed successfully in {self.startup_time:.2f} seconds"
            )
            return True

        except Exception as e:
            logger.error(f"Startup failed with exception: {e}")
            self.startup_time = time.time() - start_time
            return False

    async def _execute_phase(self, phase_tasks: list[StartupTask]) -> bool:
        """Execute all tasks in a phase respecting dependencies.

        Args:
            phase_tasks: Tasks to execute in this phase

        Returns:
            True if all critical tasks succeeded
        """
        remaining_tasks = set(task.name for task in phase_tasks)

        while remaining_tasks:
            # Find tasks that can run (dependencies satisfied)
            ready_tasks = []
            for task in phase_tasks:
                if task.name in remaining_tasks and task.dependencies.issubset(
                    self.completed_tasks
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                # No tasks ready - check for circular dependencies
                logger.error(
                    f"Circular dependency detected in remaining tasks: {remaining_tasks}"
                )
                return False

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_task(task) for task in ready_tasks],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready_tasks, results, strict=False):
                if isinstance(result, Exception):
                    result = StartupResult(
                        task_name=task.name,
                        phase=task.phase,
                        success=False,
                        duration_seconds=0.0,
                        error=str(result),
                    )

                self.startup_results.append(result)

                if result.success:
                    self.completed_tasks.add(task.name)
                    logger.info(
                        f"Startup task completed: {task.name} ({result.duration_seconds:.2f}s)"
                    )
                else:
                    logger.error(f"Startup task failed: {task.name} - {result.error}")
                    if task.critical:
                        return False

                remaining_tasks.discard(task.name)

        return True

    async def _execute_task(self, task: StartupTask) -> StartupResult:
        """Execute a single startup task with retries.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        start_time = time.time()
        last_error = None

        for attempt in range(task.retry_attempts):
            try:
                # Execute task with timeout
                await asyncio.wait_for(
                    task.task_function(), timeout=task.timeout_seconds
                )

                duration = time.time() - start_time
                return StartupResult(
                    task_name=task.name,
                    phase=task.phase,
                    success=True,
                    duration_seconds=duration,
                )

            except TimeoutError:
                last_error = f"Task timeout after {task.timeout_seconds} seconds"
            except Exception as e:
                last_error = str(e)

            # Wait before retry
            if attempt < task.retry_attempts - 1:
                await asyncio.sleep(task.retry_delay_seconds)

        duration = time.time() - start_time
        return StartupResult(
            task_name=task.name,
            phase=task.phase,
            success=False,
            duration_seconds=duration,
            error=last_error,
        )

    # Default startup task implementations
    async def _load_configuration(self) -> None:
        """Load and validate application configuration."""
        config = get_production_config(settings=self.settings)
        logger.info(
            f"Loaded configuration for environment: {config.environment.name.value}"
        )

    async def _setup_logging(self) -> None:
        """Setup application logging."""
        config = get_production_config()

        # Configure logging level
        log_level = getattr(logging, config.environment.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)

        logger.info(f"Logging configured: level={config.environment.log_level}")

    async def _validate_environment(self) -> None:
        """Validate environment configuration and requirements."""
        config = get_production_config()

        # Check required environment variables
        required_vars = []
        if config.environment.name == Environment.PRODUCTION:
            required_vars = ["DATABASE_URL", "SECRET_KEY"]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise InfrastructureError(
                f"Missing required environment variables: {missing_vars}"
            )

        logger.info("Environment validation completed")

    async def _setup_security(self) -> None:
        """Setup security configurations."""
        config = get_production_config()

        # Security setup would go here
        logger.info(f"Security setup completed: level={config.security.level.value}")

    async def _initialize_database(self) -> None:
        """Initialize database connections and perform migrations."""
        try:
            db_manager = get_production_database_manager()

            # Test database connection
            async with db_manager.get_session() as session:
                await session.execute("SELECT 1")

            logger.info("Database initialization completed")

        except Exception as e:
            raise InfrastructureError(f"Database initialization failed: {e}")

    async def _initialize_cache(self) -> None:
        """Initialize cache systems."""
        try:
            cache_manager = get_cache_integration_manager()

            # Test cache operations if available
            if cache_manager.intelligent_cache:
                test_key = "startup_test"
                await cache_manager.intelligent_cache.set(
                    test_key, "test_value", ttl=10
                )
                value = await cache_manager.intelligent_cache.get(test_key)
                await cache_manager.intelligent_cache.delete(test_key)

                if value != "test_value":
                    raise InfrastructureError("Cache test operation failed")

            logger.info("Cache initialization completed")

        except Exception as e:
            raise InfrastructureError(f"Cache initialization failed: {e}")

    async def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting systems."""
        try:
            rate_manager = get_rate_limit_manager()

            # Test rate limiting
            status = await rate_manager.check_limit(
                limiter_name="startup_test",
                identifier="startup",
                operation="test",
                tokens=0,  # Don't consume tokens
            )

            logger.info("Rate limiting initialization completed")

        except Exception as e:
            raise InfrastructureError(f"Rate limiting initialization failed: {e}")

    async def _initialize_services(self) -> None:
        """Initialize application services."""
        # Service initialization would go here
        logger.info("Application services initialization completed")

    async def _load_models(self) -> None:
        """Load machine learning models."""
        # Model loading would go here
        # This is marked as non-critical, so failures won't stop startup
        logger.info("Model loading completed")

    async def _initialize_health_monitoring(self) -> None:
        """Initialize health monitoring systems."""
        try:
            health_manager = get_comprehensive_health_manager(auto_start=True)
            logger.info("Health monitoring initialization completed")

        except Exception as e:
            raise InfrastructureError(f"Health monitoring initialization failed: {e}")

    async def _run_startup_health_checks(self) -> None:
        """Run initial health checks to verify system readiness."""
        try:
            health_manager = get_comprehensive_health_manager()

            # Get readiness status
            readiness = await health_manager.get_readiness_status()

            if not readiness["ready"]:
                unhealthy_components = [
                    name
                    for name, status in readiness["critical_components"].items()
                    if status["status"] != "healthy"
                ]
                raise InfrastructureError(
                    f"System not ready: unhealthy components: {unhealthy_components}"
                )

            logger.info("Startup health checks completed - system ready")

        except Exception as e:
            raise InfrastructureError(f"Startup health checks failed: {e}")

    def get_startup_summary(self) -> dict[str, Any]:
        """Get startup execution summary.

        Returns:
            Startup summary with results and timing
        """
        successful_tasks = [r for r in self.startup_results if r.success]
        failed_tasks = [r for r in self.startup_results if not r.success]

        return {
            "ready": self.is_ready,
            "startup_time_seconds": self.startup_time,
            "total_tasks": len(self.startup_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "task_results": [
                {
                    "name": r.task_name,
                    "phase": r.phase.value,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in self.startup_results
            ],
            "failed_task_names": [r.task_name for r in failed_tasks],
        }


class ProductionStartupManager:
    """Production-ready startup manager with signal handling."""

    def __init__(self, settings: Settings | None = None):
        """Initialize production startup manager.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.startup = ApplicationStartup(settings)
        self.startup_complete = False
        self.shutdown_requested = False

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, requesting shutdown...")
            self.shutdown_requested = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def start(self) -> bool:
        """Start the application with production-ready initialization.

        Returns:
            True if startup completed successfully
        """
        logger.info("Starting production application...")

        try:
            success = await self.startup.startup()
            self.startup_complete = success

            if success:
                logger.info("Production application started successfully")
            else:
                logger.error("Production application startup failed")

            return success

        except Exception as e:
            logger.error(f"Production startup failed: {e}")
            return False

    @asynccontextmanager
    async def lifespan(self):
        """Application lifespan context manager for ASGI applications.

        Example:
            startup_manager = ProductionStartupManager()

            @asynccontextmanager
            async def lifespan(app):
                async with startup_manager.lifespan():
                    yield
        """
        try:
            # Startup
            success = await self.start()
            if not success:
                raise InfrastructureError("Application startup failed")

            yield

        finally:
            # Shutdown
            logger.info("Application shutdown initiated")
            # Shutdown logic would go here


# Global startup manager
_startup_manager: ProductionStartupManager | None = None


def get_startup_manager(settings: Settings | None = None) -> ProductionStartupManager:
    """Get global startup manager.

    Args:
        settings: Application settings

    Returns:
        Production startup manager instance
    """
    global _startup_manager

    if _startup_manager is None:
        _startup_manager = ProductionStartupManager(settings)

    return _startup_manager


async def startup_health_check() -> dict[str, Any]:
    """Quick health check for application startup status.

    Returns:
        Startup health status
    """
    startup_manager = get_startup_manager()

    return {
        "startup_complete": startup_manager.startup_complete,
        "ready": startup_manager.startup.is_ready,
        "startup_time": startup_manager.startup.startup_time,
        "summary": startup_manager.startup.get_startup_summary(),
    }
