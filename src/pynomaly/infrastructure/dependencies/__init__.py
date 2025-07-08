"""
Forward-reference-free dependency injection system for FastAPI.

This module provides a clean way to handle dependencies without circular imports
by using a registry pattern that allows late binding of dependencies.
"""

from .wrapper import (
    DependencyWrapper,
    DependencyRegistry,
    register_dependency,
    register_dependency_provider,
    get_dependency,
    is_dependency_available,
    clear_dependencies,
    initialize_dependencies,
    dependency_context,
    # Common service wrappers
    auth_service,
    user_service,
    detection_service,
    model_service,
    database_service,
    cache_service,
    metrics_service,
)

__all__ = [
    "DependencyWrapper",
    "DependencyRegistry",
    "register_dependency",
    "register_dependency_provider",
    "get_dependency",
    "is_dependency_available",
    "clear_dependencies",
    "initialize_dependencies",
    "dependency_context",
    "auth_service",
    "user_service",
    "detection_service",
    "model_service",
    "database_service",
    "cache_service",
    "metrics_service",
]
