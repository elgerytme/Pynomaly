"""Enterprise Core Framework.

A comprehensive core framework package providing domain abstractions,
dependency injection, and clean architecture patterns for enterprise applications.
"""

from .domain import AggregateRoot, BaseEntity, BaseValueObject, DomainEvent
from .exceptions import (
    ConfigurationError,
    DomainError,
    EnterpriseError,
    InfrastructureError,
)
from .infrastructure import (
    ConfigurationManager,
    Container,
    FeatureFlagManager,
    ServiceRegistry,
)
from .protocols import Cache, EventBus, Logger, MessageQueue, Repository

__version__ = "0.1.0"
__all__ = [
    # Domain abstractions
    "BaseEntity",
    "BaseValueObject",
    "DomainEvent",
    "AggregateRoot",
    # Infrastructure
    "Container",
    "ServiceRegistry",
    "FeatureFlagManager",
    "ConfigurationManager",
    # Protocols
    "Repository",
    "EventBus",
    "Logger",
    "Cache",
    "MessageQueue",
    # Exceptions
    "EnterpriseError",
    "DomainError",
    "InfrastructureError",
    "ConfigurationError",
]
