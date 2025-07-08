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

# Import setup functionality
from .setup import setup_dependencies, DependencySetup

# Import test utilities (optional)
try:
    from .test_utils import (
        DependencyValidator,
        test_dependency_context,
        create_mock_dependencies,
        setup_test_dependencies,
        validate_standard_dependencies,
        run_dependency_health_check,
    )
    TEST_UTILS_AVAILABLE = True
except ImportError:
    TEST_UTILS_AVAILABLE = False

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
    "setup_dependencies",
    "DependencySetup",
]

# Add test utilities to __all__ if available
if TEST_UTILS_AVAILABLE:
    __all__.extend([
        "DependencyValidator",
        "test_dependency_context",
        "create_mock_dependencies",
        "setup_test_dependencies",
        "validate_standard_dependencies",
        "run_dependency_health_check",
    ])
