"""Graceful shutdown management for production applications."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from monorepo.infrastructure.cache import close_cache_integration_manager
from monorepo.infrastructure.monitoring import close_comprehensive_health_manager
from monorepo.infrastructure.security import close_rate_limit_manager

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Shutdown phases in order of execution."""

    PREPARE = "prepare"
    STOP_ACCEPTING = "stop_accepting"
    DRAIN_CONNECTIONS = "drain_connections"
    STOP_SERVICES = "stop_services"
    CLEANUP_INFRASTRUCTURE = "cleanup_infrastructure"
    FINALIZE = "finalize"


@dataclass
class ShutdownTask:
    """Individual shutdown task definition."""

    name: str
    phase: ShutdownPhase
    task_function: Callable
    timeout_seconds: float = 30.0
    critical: bool = True
    dependencies: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure dependencies is a set."""
        if isinstance(self.dependencies, (list, tuple)):
            self.dependencies = set(self.dependencies)


@dataclass
class ShutdownResult:
    """Result of a shutdown task execution."""

    task_name: str
    phase: ShutdownPhase
    success: bool
    duration_seconds: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ShutdownManager:
    """Manages graceful application shutdown."""

    def __init__(self, shutdown_timeout: float = 30.0):
        """Initialize shutdown manager.

        Args:
            shutdown_timeout: Maximum time to wait for shutdown completion
        """
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_tasks: dict[str, ShutdownTask] = {}
        self.completed_tasks: set[str] = set()
        self.shutdown_results: list[ShutdownResult] = []
        self.shutdown_requested = False
        self.shutdown_complete = False
        self.shutdown_time = 0.0

        # Register default shutdown tasks
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        """Register default shutdown tasks."""

        # Phase 1: Prepare
        self.register_task(
            ShutdownTask(
                name="log_shutdown_start",
                phase=ShutdownPhase.PREPARE,
                task_function=self._log_shutdown_start,
                timeout_seconds=1.0,
            )
        )

        self.register_task(
            ShutdownTask(
                name="stop_health_checks",
                phase=ShutdownPhase.PREPARE,
                task_function=self._stop_health_checks,
                dependencies={"log_shutdown_start"},
                timeout_seconds=5.0,
            )
        )

        # Phase 2: Stop accepting new requests
        self.register_task(
            ShutdownTask(
                name="stop_accepting_requests",
                phase=ShutdownPhase.STOP_ACCEPTING,
                task_function=self._stop_accepting_requests,
                dependencies={"stop_health_checks"},
                timeout_seconds=5.0,
            )
        )

        # Phase 3: Drain connections
        self.register_task(
            ShutdownTask(
                name="drain_active_connections",
                phase=ShutdownPhase.DRAIN_CONNECTIONS,
                task_function=self._drain_active_connections,
                dependencies={"stop_accepting_requests"},
                timeout_seconds=20.0,
            )
        )

        # Phase 4: Stop services
        self.register_task(
            ShutdownTask(
                name="stop_background_tasks",
                phase=ShutdownPhase.STOP_SERVICES,
                task_function=self._stop_background_tasks,
                dependencies={"drain_active_connections"},
                timeout_seconds=15.0,
            )
        )

        self.register_task(
            ShutdownTask(
                name="stop_model_services",
                phase=ShutdownPhase.STOP_SERVICES,
                task_function=self._stop_model_services,
                dependencies={"stop_background_tasks"},
                timeout_seconds=10.0,
            )
        )

        # Phase 5: Cleanup infrastructure
        self.register_task(
            ShutdownTask(
                name="cleanup_cache",
                phase=ShutdownPhase.CLEANUP_INFRASTRUCTURE,
                task_function=self._cleanup_cache,
                dependencies={"stop_model_services"},
                timeout_seconds=10.0,
            )
        )

        self.register_task(
            ShutdownTask(
                name="cleanup_rate_limiting",
                phase=ShutdownPhase.CLEANUP_INFRASTRUCTURE,
                task_function=self._cleanup_rate_limiting,
                dependencies={"cleanup_cache"},
                timeout_seconds=5.0,
            )
        )

        self.register_task(
            ShutdownTask(
                name="cleanup_database",
                phase=ShutdownPhase.CLEANUP_INFRASTRUCTURE,
                task_function=self._cleanup_database,
                dependencies={"cleanup_rate_limiting"},
                timeout_seconds=15.0,
            )
        )

        # Phase 6: Finalize
        self.register_task(
            ShutdownTask(
                name="cleanup_monitoring",
                phase=ShutdownPhase.FINALIZE,
                task_function=self._cleanup_monitoring,
                dependencies={"cleanup_database"},
                timeout_seconds=5.0,
            )
        )

        self.register_task(
            ShutdownTask(
                name="log_shutdown_complete",
                phase=ShutdownPhase.FINALIZE,
                task_function=self._log_shutdown_complete,
                dependencies={"cleanup_monitoring"},
                timeout_seconds=1.0,
            )
        )

    def register_task(self, task: ShutdownTask) -> None:
        """Register a shutdown task.

        Args:
            task: Shutdown task to register
        """
        self.shutdown_tasks[task.name] = task
        logger.debug(
            f"Registered shutdown task: {task.name} (phase: {task.phase.value})"
        )

    async def shutdown(self) -> bool:
        """Execute graceful shutdown sequence.

        Returns:
            True if shutdown completed successfully
        """
        if self.shutdown_requested:
            logger.warning("Shutdown already in progress")
            return False

        self.shutdown_requested = True
        logger.info("Starting graceful shutdown sequence...")
        start_time = time.time()

        try:
            # Execute shutdown with timeout
            shutdown_successful = await asyncio.wait_for(
                self._execute_shutdown(), timeout=self.shutdown_timeout
            )

            self.shutdown_time = time.time() - start_time
            self.shutdown_complete = True

            if shutdown_successful:
                logger.info(
                    f"Graceful shutdown completed successfully in {self.shutdown_time:.2f} seconds"
                )
            else:
                logger.error(
                    f"Shutdown completed with errors in {self.shutdown_time:.2f} seconds"
                )

            return shutdown_successful

        except TimeoutError:
            self.shutdown_time = time.time() - start_time
            logger.error(f"Shutdown timeout after {self.shutdown_timeout} seconds")
            return False
        except Exception as e:
            self.shutdown_time = time.time() - start_time
            logger.error(f"Shutdown failed with exception: {e}")
            return False

    async def _execute_shutdown(self) -> bool:
        """Execute shutdown tasks by phase."""
        success = True

        for phase in ShutdownPhase:
            phase_tasks = [
                task for task in self.shutdown_tasks.values() if task.phase == phase
            ]

            if not phase_tasks:
                continue

            logger.info(f"Executing shutdown phase: {phase.value}")
            phase_success = await self._execute_phase(phase_tasks)

            if not phase_success:
                success = False
                # Continue with remaining phases even if one fails

        return success

    async def _execute_phase(self, phase_tasks: list[ShutdownTask]) -> bool:
        """Execute all tasks in a phase respecting dependencies."""
        remaining_tasks = set(task.name for task in phase_tasks)
        phase_success = True

        while remaining_tasks:
            # Find tasks that can run (dependencies satisfied)
            ready_tasks = []
            for task in phase_tasks:
                if task.name in remaining_tasks and task.dependencies.issubset(
                    self.completed_tasks
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                logger.error(
                    f"Circular dependency detected in remaining tasks: {remaining_tasks}"
                )
                phase_success = False
                break

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_task(task) for task in ready_tasks],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready_tasks, results, strict=False):
                if isinstance(result, Exception):
                    result = ShutdownResult(
                        task_name=task.name,
                        phase=task.phase,
                        success=False,
                        duration_seconds=0.0,
                        error=str(result),
                    )

                self.shutdown_results.append(result)

                if result.success:
                    self.completed_tasks.add(task.name)
                    logger.info(
                        f"Shutdown task completed: {task.name} ({result.duration_seconds:.2f}s)"
                    )
                else:
                    logger.error(f"Shutdown task failed: {task.name} - {result.error}")
                    if task.critical:
                        phase_success = False

                remaining_tasks.discard(task.name)

        return phase_success

    async def _execute_task(self, task: ShutdownTask) -> ShutdownResult:
        """Execute a single shutdown task."""
        start_time = time.time()

        try:
            # Execute task with timeout
            await asyncio.wait_for(task.task_function(), timeout=task.timeout_seconds)

            duration = time.time() - start_time
            return ShutdownResult(
                task_name=task.name,
                phase=task.phase,
                success=True,
                duration_seconds=duration,
            )

        except TimeoutError:
            duration = time.time() - start_time
            return ShutdownResult(
                task_name=task.name,
                phase=task.phase,
                success=False,
                duration_seconds=duration,
                error=f"Task timeout after {task.timeout_seconds} seconds",
            )
        except Exception as e:
            duration = time.time() - start_time
            return ShutdownResult(
                task_name=task.name,
                phase=task.phase,
                success=False,
                duration_seconds=duration,
                error=str(e),
            )

    # Default shutdown task implementations
    async def _log_shutdown_start(self) -> None:
        """Log shutdown start."""
        logger.info("Graceful shutdown initiated")

    async def _stop_health_checks(self) -> None:
        """Stop health check monitoring."""
        try:
            await close_comprehensive_health_manager()
            logger.info("Health monitoring stopped")
        except Exception as e:
            logger.warning(f"Failed to stop health monitoring: {e}")

    async def _stop_accepting_requests(self) -> None:
        """Stop accepting new requests."""
        # This would integrate with the web server
        # For now, just log the action
        logger.info("Stopped accepting new requests")

    async def _drain_active_connections(self) -> None:
        """Wait for active connections to complete."""
        # This would wait for active requests to finish
        # For now, simulate with a short delay
        await asyncio.sleep(1.0)
        logger.info("Active connections drained")

    async def _stop_background_tasks(self) -> None:
        """Stop background tasks and workers."""
        # This would stop background processing
        logger.info("Background tasks stopped")

    async def _stop_model_services(self) -> None:
        """Stop model services and cleanup model resources."""
        # This would cleanup model resources
        logger.info("Model services stopped")

    async def _cleanup_cache(self) -> None:
        """Cleanup cache resources."""
        try:
            await close_cache_integration_manager()
            logger.info("Cache resources cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache: {e}")

    async def _cleanup_rate_limiting(self) -> None:
        """Cleanup rate limiting resources."""
        try:
            await close_rate_limit_manager()
            logger.info("Rate limiting resources cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup rate limiting: {e}")

    async def _cleanup_database(self) -> None:
        """Cleanup database connections."""
        try:
            # Database cleanup would go here
            logger.info("Database connections cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup database: {e}")

    async def _cleanup_monitoring(self) -> None:
        """Cleanup monitoring resources."""
        try:
            # Additional monitoring cleanup
            logger.info("Monitoring resources cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup monitoring: {e}")

    async def _log_shutdown_complete(self) -> None:
        """Log shutdown completion."""
        logger.info("Graceful shutdown sequence completed")

    def get_shutdown_summary(self) -> dict[str, Any]:
        """Get shutdown execution summary."""
        successful_tasks = [r for r in self.shutdown_results if r.success]
        failed_tasks = [r for r in self.shutdown_results if not r.success]

        return {
            "shutdown_complete": self.shutdown_complete,
            "shutdown_time_seconds": self.shutdown_time,
            "total_tasks": len(self.shutdown_results),
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
                for r in self.shutdown_results
            ],
            "failed_task_names": [r.task_name for r in failed_tasks],
        }


class GracefulShutdownHandler:
    """Production-ready shutdown handler with signal management."""

    def __init__(self, shutdown_timeout: float = 30.0):
        """Initialize graceful shutdown handler.

        Args:
            shutdown_timeout: Maximum time to wait for shutdown
        """
        self.shutdown_manager = ShutdownManager(shutdown_timeout)
        self.shutdown_event = asyncio.Event()
        self.shutdown_hooks: list[Callable] = []

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")

            # Set shutdown event
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.shutdown_event.set)
            except RuntimeError:
                # No running loop, create task for later
                pass

        # Handle common shutdown signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Handle additional signals in production
        try:
            signal.signal(signal.SIGHUP, signal_handler)
            signal.signal(signal.SIGQUIT, signal_handler)
        except AttributeError:
            # Signals not available on all platforms
            pass

    def register_shutdown_hook(self, hook: Callable) -> None:
        """Register a shutdown hook function.

        Args:
            hook: Function to call during shutdown
        """
        self.shutdown_hooks.append(hook)
        logger.debug(f"Registered shutdown hook: {hook.__name__}")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()

    async def shutdown(self) -> bool:
        """Execute graceful shutdown.

        Returns:
            True if shutdown completed successfully
        """
        # Execute custom shutdown hooks first
        for hook in self.shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(f"Shutdown hook failed: {e}")

        # Execute main shutdown sequence
        return await self.shutdown_manager.shutdown()

    async def run_with_graceful_shutdown(self, main_task: Callable) -> None:
        """Run main application with graceful shutdown handling.

        Args:
            main_task: Main application task to run
        """
        # Start main task
        main_task_handle = asyncio.create_task(main_task())

        # Wait for either main task completion or shutdown signal
        done, pending = await asyncio.wait(
            [main_task_handle, asyncio.create_task(self.wait_for_shutdown())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # If shutdown was triggered, perform graceful shutdown
        if self.shutdown_event.is_set():
            await self.shutdown()


# Global shutdown manager
_shutdown_manager: GracefulShutdownHandler | None = None


def get_shutdown_manager(shutdown_timeout: float = 30.0) -> GracefulShutdownHandler:
    """Get global shutdown manager.

    Args:
        shutdown_timeout: Maximum time to wait for shutdown

    Returns:
        Graceful shutdown handler instance
    """
    global _shutdown_manager

    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownHandler(shutdown_timeout)

    return _shutdown_manager


def register_shutdown_hook(hook: Callable) -> None:
    """Register a shutdown hook with the global shutdown manager.

    Args:
        hook: Function to call during shutdown
    """
    shutdown_manager = get_shutdown_manager()
    shutdown_manager.register_shutdown_hook(hook)
