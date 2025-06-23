"""Graceful shutdown service for proper resource cleanup."""

from __future__ import annotations

import asyncio
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ShutdownPhase(Enum):
    """Shutdown phases."""
    STOP_ACCEPTING_REQUESTS = "stop_accepting_requests"
    DRAIN_CONNECTIONS = "drain_connections"
    CLEANUP_RESOURCES = "cleanup_resources"
    FINALIZE = "finalize"


@dataclass
class ShutdownHandler:
    """Shutdown handler configuration."""
    name: str
    phase: ShutdownPhase
    handler: Callable[[], Any]
    timeout: float = 30.0
    is_async: bool = False


class ShutdownService:
    """Service for managing graceful application shutdown."""
    
    def __init__(self, shutdown_timeout: float = 60.0):
        """Initialize shutdown service.
        
        Args:
            shutdown_timeout: Total timeout for shutdown process in seconds
        """
        self.shutdown_timeout = shutdown_timeout
        self._handlers: Dict[ShutdownPhase, List[ShutdownHandler]] = {
            phase: [] for phase in ShutdownPhase
        }
        self._shutdown_event = threading.Event()
        self._shutdown_in_progress = False
        self._start_time = time.time()
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def register_handler(
        self,
        name: str,
        handler: Callable[[], Any],
        phase: ShutdownPhase = ShutdownPhase.CLEANUP_RESOURCES,
        timeout: float = 30.0,
        is_async: bool = False
    ) -> None:
        """Register a shutdown handler.
        
        Args:
            name: Human-readable name for the handler
            handler: Function to call during shutdown
            phase: Shutdown phase when to call the handler
            timeout: Maximum time to wait for handler completion
            is_async: Whether the handler is async
        """
        shutdown_handler = ShutdownHandler(
            name=name,
            phase=phase,
            handler=handler,
            timeout=timeout,
            is_async=is_async
        )
        
        self._handlers[phase].append(shutdown_handler)
        logger.info(
            "Registered shutdown handler",
            name=name,
            phase=phase.value,
            timeout=timeout
        )
    
    def _register_signal_handlers(self) -> None:
        """Register system signal handlers."""
        def signal_handler(signum: int, frame: Any) -> None:
            signal_name = signal.Signals(signum).name
            logger.info("Received shutdown signal", signal=signal_name)
            self.initiate_shutdown()
        
        # Register for common shutdown signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, signal_handler)
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown process."""
        if self._shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return
        
        self._shutdown_in_progress = True
        self._shutdown_event.set()
        
        logger.info(
            "Initiating graceful shutdown",
            timeout=self.shutdown_timeout,
            uptime_seconds=time.time() - self._start_time
        )
        
        # Run shutdown in background thread to avoid blocking
        shutdown_thread = threading.Thread(
            target=self._execute_shutdown,
            name="shutdown-thread"
        )
        shutdown_thread.start()
    
    def _execute_shutdown(self) -> None:
        """Execute the shutdown process."""
        start_time = time.time()
        
        try:
            # Execute handlers in each phase
            for phase in ShutdownPhase:
                if not self._handlers[phase]:
                    continue
                
                logger.info(f"Executing shutdown phase: {phase.value}")
                phase_start = time.time()
                
                # Execute all handlers in this phase
                for handler in self._handlers[phase]:
                    self._execute_handler(handler)
                
                phase_duration = time.time() - phase_start
                logger.info(
                    f"Completed shutdown phase: {phase.value}",
                    duration=phase_duration,
                    handlers_count=len(self._handlers[phase])
                )
            
            total_duration = time.time() - start_time
            logger.info(
                "Graceful shutdown completed successfully",
                total_duration=total_duration
            )
            
        except Exception as e:
            logger.error(
                "Error during graceful shutdown",
                error=str(e),
                duration=time.time() - start_time
            )
        
        finally:
            # Final cleanup
            logger.info("Application shutdown complete")
    
    def _execute_handler(self, handler: ShutdownHandler) -> None:
        """Execute a single shutdown handler."""
        handler_start = time.time()
        
        try:
            logger.info(f"Executing shutdown handler: {handler.name}")
            
            if handler.is_async:
                # Run async handler
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task = loop.create_task(handler.handler())
                    loop.run_until_complete(
                        asyncio.wait_for(task, timeout=handler.timeout)
                    )
                finally:
                    loop.close()
            else:
                # Run sync handler with timeout (basic implementation)
                handler.handler()
            
            handler_duration = time.time() - handler_start
            logger.info(
                f"Shutdown handler completed: {handler.name}",
                duration=handler_duration
            )
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Shutdown handler timed out: {handler.name}",
                timeout=handler.timeout
            )
        except Exception as e:
            logger.error(
                f"Shutdown handler failed: {handler.name}",
                error=str(e),
                duration=time.time() - handler_start
            )
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to be initiated.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if shutdown was initiated, False if timeout
        """
        return self._shutdown_event.wait(timeout)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is currently in progress."""
        return self._shutdown_in_progress
    
    def register_database_cleanup(self, database_manager: Any) -> None:
        """Register database cleanup handler."""
        def cleanup_database():
            try:
                if hasattr(database_manager, 'close_all_sessions'):
                    database_manager.close_all_sessions()
                if hasattr(database_manager, 'dispose'):
                    database_manager.dispose()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error("Failed to cleanup database", error=str(e))
        
        self.register_handler(
            name="database_cleanup",
            handler=cleanup_database,
            phase=ShutdownPhase.CLEANUP_RESOURCES,
            timeout=10.0
        )
    
    def register_cache_cleanup(self, cache_client: Any) -> None:
        """Register cache cleanup handler."""
        def cleanup_cache():
            try:
                if hasattr(cache_client, 'close'):
                    cache_client.close()
                elif hasattr(cache_client, 'disconnect'):
                    cache_client.disconnect()
                logger.info("Cache connections closed")
            except Exception as e:
                logger.error("Failed to cleanup cache", error=str(e))
        
        self.register_handler(
            name="cache_cleanup",
            handler=cleanup_cache,
            phase=ShutdownPhase.CLEANUP_RESOURCES,
            timeout=5.0
        )
    
    def register_telemetry_cleanup(self, telemetry_service: Any) -> None:
        """Register telemetry service cleanup handler."""
        def cleanup_telemetry():
            try:
                if hasattr(telemetry_service, 'shutdown'):
                    telemetry_service.shutdown()
                elif hasattr(telemetry_service, 'close'):
                    telemetry_service.close()
                logger.info("Telemetry service shutdown")
            except Exception as e:
                logger.error("Failed to cleanup telemetry", error=str(e))
        
        self.register_handler(
            name="telemetry_cleanup",
            handler=cleanup_telemetry,
            phase=ShutdownPhase.FINALIZE,
            timeout=5.0
        )
    
    def register_server_stop(self, server: Any) -> None:
        """Register server stop handler."""
        async def stop_server():
            try:
                if hasattr(server, 'shutdown'):
                    await server.shutdown()
                elif hasattr(server, 'close'):
                    await server.close()
                logger.info("Server stopped")
            except Exception as e:
                logger.error("Failed to stop server", error=str(e))
        
        self.register_handler(
            name="server_stop",
            handler=stop_server,
            phase=ShutdownPhase.STOP_ACCEPTING_REQUESTS,
            timeout=15.0,
            is_async=True
        )
    
    def register_background_tasks_cleanup(self, task_manager: Any) -> None:
        """Register background tasks cleanup handler."""
        async def cleanup_tasks():
            try:
                if hasattr(task_manager, 'cancel_all'):
                    await task_manager.cancel_all()
                elif hasattr(task_manager, 'shutdown'):
                    await task_manager.shutdown()
                logger.info("Background tasks cleaned up")
            except Exception as e:
                logger.error("Failed to cleanup background tasks", error=str(e))
        
        self.register_handler(
            name="background_tasks_cleanup", 
            handler=cleanup_tasks,
            phase=ShutdownPhase.DRAIN_CONNECTIONS,
            timeout=20.0,
            is_async=True
        )
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        return {
            "shutdown_requested": self.is_shutdown_requested(),
            "shutdown_in_progress": self.is_shutdown_in_progress(),
            "registered_handlers": {
                phase.value: len(handlers)
                for phase, handlers in self._handlers.items()
            },
            "uptime_seconds": time.time() - self._start_time,
            "shutdown_timeout": self.shutdown_timeout
        }


# Global shutdown service instance
_shutdown_service: Optional[ShutdownService] = None


def get_shutdown_service() -> ShutdownService:
    """Get the global shutdown service instance."""
    global _shutdown_service
    if _shutdown_service is None:
        _shutdown_service = ShutdownService()
    return _shutdown_service


def initiate_shutdown() -> None:
    """Initiate graceful shutdown."""
    get_shutdown_service().initiate_shutdown()


def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Wait for shutdown to be initiated."""
    return get_shutdown_service().wait_for_shutdown(timeout)