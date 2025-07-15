"""Async utilities for CLI to handle event loop compatibility."""

import asyncio
import functools
import warnings
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_async_safely(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine safely, handling event loop conflicts.
    
    This function detects if an event loop is already running and uses
    appropriate methods to execute the coroutine without conflicts.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine execution
        
    Raises:
        RuntimeError: If there's an unrecoverable event loop error
    """
    try:
        # First, try the standard approach
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # We're in an existing event loop, try alternative approaches
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                
                # Create a new task in the existing loop
                task = loop.create_task(coro)
                
                # Wait for completion using a different approach
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
                
            except RuntimeError:
                # If that fails, try using nest_asyncio if available
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(coro)
                except ImportError:
                    # Fall back to creating a new event loop in a thread
                    import threading
                    import concurrent.futures
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(coro)
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result()
        else:
            # Re-raise other RuntimeErrors
            raise


def async_command(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator to make async functions CLI-compatible.
    
    This decorator allows async functions to be used as CLI commands
    by automatically handling event loop management.
    
    Args:
        func: The async function to wrap
        
    Returns:
        A synchronous wrapper function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)
        return run_async_safely(coro)
    
    return wrapper


def handle_event_loop_warnings():
    """
    Configure warnings to handle event loop related warnings gracefully.
    
    This function sets up warning filters to suppress or handle
    common event loop warnings that don't affect functionality.
    """
    # Filter out common asyncio warnings that don't affect CLI functionality
    warnings.filterwarnings(
        "ignore",
        message="coroutine.*was never awaited",
        category=RuntimeWarning
    )
    
    warnings.filterwarnings(
        "ignore", 
        message=".*loop is already running",
        category=RuntimeWarning
    )


def ensure_event_loop_policy():
    """
    Ensure a proper event loop policy is set for CLI operations.
    
    This function sets up the appropriate event loop policy for
    different platforms to avoid compatibility issues.
    """
    import sys
    
    if sys.platform == "win32":
        # Use ProactorEventLoop on Windows for better compatibility
        if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    else:
        # Use default policy on Unix systems
        if hasattr(asyncio, "DefaultEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


def create_cli_container_safely():
    """
    Create CLI container with proper async handling.
    
    This function ensures that the CLI container is created
    with proper event loop management.
    """
    from pynomaly.presentation.cli.container import get_cli_container
    
    # Initialize event loop policy
    ensure_event_loop_policy()
    
    # Handle warnings
    handle_event_loop_warnings()
    
    # Return container
    return get_cli_container()


class AsyncCLIRunner:
    """
    A runner class for executing async CLI operations safely.
    
    This class provides methods to execute async operations
    in CLI context with proper error handling and event loop management.
    """
    
    def __init__(self):
        self.loop = None
        self._setup_event_loop()
    
    def _setup_event_loop(self):
        """Set up the event loop for CLI operations."""
        ensure_event_loop_policy()
        handle_event_loop_warnings()
    
    def run(self, coro: Awaitable[T]) -> T:
        """
        Run a coroutine safely.
        
        Args:
            coro: The coroutine to execute
            
        Returns:
            The result of the coroutine
        """
        return run_async_safely(coro)
    
    def run_use_case(self, use_case: Any, request: Any) -> Any:
        """
        Run a use case execute method safely.
        
        Args:
            use_case: The use case instance
            request: The request object
            
        Returns:
            The response from the use case
        """
        return self.run(use_case.execute(request))


# Global runner instance for CLI operations
cli_runner = AsyncCLIRunner()