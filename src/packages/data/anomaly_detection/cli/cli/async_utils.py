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
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Event loop is already running, use get_event_loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an active event loop, need to use different approach
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(coro)
            else:
                # Loop exists but isn't running
                return loop.run_until_complete(coro)
        else:
            # Different runtime error, re-raise
            raise


def async_to_sync(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator to convert async functions to sync for CLI use.
    
    Args:
        func: The async function to wrap
        
    Returns:
        A sync version of the function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_async_safely(func(*args, **kwargs))
    
    return wrapper


class CLIRunner:
    """CLI runner for async operations."""
    
    def __init__(self):
        self._loop = None
    
    def run_use_case(self, use_case, request):
        """Run a use case synchronously."""
        if hasattr(use_case, 'execute'):
            if asyncio.iscoroutinefunction(use_case.execute):
                return run_async_safely(use_case.execute(request))
            else:
                return use_case.execute(request)
        else:
            raise ValueError(f"Use case {use_case} does not have an execute method")
    
    def run_coroutine(self, coro: Awaitable[T]) -> T:
        """Run a coroutine synchronously."""
        return run_async_safely(coro)


# Global CLI runner instance
cli_runner = CLIRunner()