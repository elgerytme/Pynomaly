"""
Lightweight wrapper around FastAPI's Depends that avoids circular imports.

This module provides a forward-reference-free dependency system that allows
declaring dependencies without type hints in router files, then injecting
the fully typed callables during startup to avoid circular import issues.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager
import logging

from fastapi import Depends, HTTPException, status

logger = logging.getLogger(__name__)


class DependencyRegistry:
    """Registry for managing dependency injection without forward references."""
    
    def __init__(self):
        self._dependencies: Dict[str, Any] = {}
        self._providers: Dict[str, Callable[[], Any]] = {}
        self._initialized = False
    
    def register_provider(self, key: str, provider: Callable[[], Any]) -> None:
        """Register a dependency provider function.
        
        Args:
            key: Unique identifier for the dependency
            provider: Function that returns the dependency instance
        """
        self._providers[key] = provider
        logger.debug(f"Registered dependency provider: {key}")
    
    def register_instance(self, key: str, instance: Any) -> None:
        """Register a dependency instance directly.
        
        Args:
            key: Unique identifier for the dependency
            instance: The dependency instance
        """
        self._dependencies[key] = instance
        logger.debug(f"Registered dependency instance: {key}")
    
    def get_dependency(self, key: str) -> Any:
        """Get a dependency by key.
        
        Args:
            key: Unique identifier for the dependency
            
        Returns:
            The dependency instance
            
        Raises:
            HTTPException: If dependency is not found
        """
        # First check for direct instances
        if key in self._dependencies:
            return self._dependencies[key]
        
        # Then check for providers
        if key in self._providers:
            provider = self._providers[key]
            try:
                instance = provider()
                # Cache the instance for future use
                self._dependencies[key] = instance
                return instance
            except Exception as e:
                logger.error(f"Error creating dependency {key}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Dependency {key} not available"
                )
        
        logger.warning(f"Dependency not found: {key}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dependency {key} not registered"
        )
    
    def is_available(self, key: str) -> bool:
        """Check if a dependency is available.
        
        Args:
            key: Unique identifier for the dependency
            
        Returns:
            True if dependency is available, False otherwise
        """
        return key in self._dependencies or key in self._providers
    
    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._dependencies.clear()
        self._providers.clear()
        self._initialized = False
        logger.debug("Cleared all dependencies")
    
    def set_initialized(self) -> None:
        """Mark the registry as initialized."""
        self._initialized = True
        logger.info("Dependency registry initialized")
    
    @property
    def initialized(self) -> bool:
        """Check if the registry is initialized."""
        return self._initialized


# Global registry instance
_registry = DependencyRegistry()


class DependencyWrapper:
    """Wrapper around FastAPI's Depends that provides forward-reference-free dependency injection."""
    
    def __init__(self, dependency_key: str, optional: bool = False):
        """Initialize the dependency wrapper.
        
        Args:
            dependency_key: Unique identifier for the dependency
            optional: If True, return None instead of raising exception if not found
        """
        self.dependency_key = dependency_key
        self.optional = optional
    
    def __call__(self) -> Any:
        """Create a FastAPI dependency from the registered dependency.
        
        Returns:
            FastAPI Depends instance
        """
        def _get_dependency():
            try:
                return _registry.get_dependency(self.dependency_key)
            except HTTPException:
                if self.optional:
                    return None
                raise
        
        return Depends(_get_dependency)


def register_dependency(key: str, instance: Any) -> None:
    """Register a dependency instance.
    
    Args:
        key: Unique identifier for the dependency
        instance: The dependency instance
    """
    _registry.register_instance(key, instance)


def register_dependency_provider(key: str, provider: Callable[[], Any]) -> None:
    """Register a dependency provider function.
    
    Args:
        key: Unique identifier for the dependency
        provider: Function that returns the dependency instance
    """
    _registry.register_provider(key, provider)


def get_dependency(key: str) -> Any:
    """Get a dependency by key.
    
    Args:
        key: Unique identifier for the dependency
        
    Returns:
        The dependency instance
    """
    return _registry.get_dependency(key)


def is_dependency_available(key: str) -> bool:
    """Check if a dependency is available.
    
    Args:
        key: Unique identifier for the dependency
        
    Returns:
        True if dependency is available, False otherwise
    """
    return _registry.is_available(key)


def clear_dependencies() -> None:
    """Clear all registered dependencies."""
    _registry.clear()


def initialize_dependencies() -> None:
    """Mark the dependency registry as initialized."""
    _registry.set_initialized()


@contextmanager
def dependency_context():
    """Context manager for dependency lifecycle management."""
    try:
        yield _registry
    finally:
        # Cleanup is handled by the application lifecycle
        pass


# Common dependency wrappers for typical use cases
def auth_service() -> DependencyWrapper:
    """Get the authentication service dependency."""
    return DependencyWrapper("auth_service", optional=True)


def user_service() -> DependencyWrapper:
    """Get the user service dependency."""
    return DependencyWrapper("user_service", optional=True)


def detection_service() -> DependencyWrapper:
    """Get the detection service dependency."""
    return DependencyWrapper("detection_service")


def model_service() -> DependencyWrapper:
    """Get the model service dependency."""
    return DependencyWrapper("model_service")


def database_service() -> DependencyWrapper:
    """Get the database service dependency."""
    return DependencyWrapper("database_service")


def cache_service() -> DependencyWrapper:
    """Get the cache service dependency."""
    return DependencyWrapper("cache_service", optional=True)


def metrics_service() -> DependencyWrapper:
    """Get the metrics service dependency."""
    return DependencyWrapper("metrics_service", optional=True)
