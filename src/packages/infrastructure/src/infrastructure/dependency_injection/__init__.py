"""Dependency injection infrastructure for service management.

This module provides dependency injection capabilities to decouple packages
and enable proper inversion of control patterns.
"""

from .registry import ServiceRegistry, get_registry
from .container import DIContainer, get_container
from .decorators import inject, injectable

__all__ = [
    "ServiceRegistry",
    "get_registry",
    "DIContainer", 
    "get_container",
    "inject",
    "injectable"
]