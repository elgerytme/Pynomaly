"""
Shared package for common infrastructure and utilities.

This package provides shared components that can be used across all
packages in the monorepo while maintaining proper architectural boundaries.
"""

__version__ = "0.1.0"

# Existing shared components
try:
    from .types import Result, Success, Failure, Optional, Paginated, ValidationResult
    from .value_objects import Identifier, Email, Timestamp, Money
    from .base_classes import Entity, ValueObject, DomainEvent, UseCase
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False

# New infrastructure components
from .event_bus import (
    DistributedEventBus,
    AsyncEventHandler,
    event_handler,
    get_event_bus,
    publish_event,
    subscribe_to_event,
)

from .dependency_injection import (
    DIContainer,
    LifecycleScope,
    InjectionError,
    ScopeContext,
    inject,
    injectable,
    get_container,
    configure_container,
    register_repository,
    register_service,
    register_cache,
    register_health_check,
)

# Build __all__ dynamically based on available components
__all__ = [
    # Event Bus
    "DistributedEventBus",
    "AsyncEventHandler", 
    "event_handler",
    "get_event_bus",
    "publish_event",
    "subscribe_to_event",
    
    # Dependency Injection
    "DIContainer",
    "LifecycleScope",
    "InjectionError",
    "ScopeContext",
    "inject",
    "injectable",
    "get_container",
    "configure_container",
    "register_repository",
    "register_service",
    "register_cache",
    "register_health_check",
]

# Add legacy components if available
if LEGACY_COMPONENTS_AVAILABLE:
    __all__.extend([
        # Types
        "Result",
        "Success", 
        "Failure",
        "Optional",
        "Paginated",
        "ValidationResult",
        # Value Objects
        "Identifier",
        "Email",
        "Timestamp", 
        "Money",
        # Base Classes
        "Entity",
        "ValueObject",
        "DomainEvent",
        "UseCase",
    ])