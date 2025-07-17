"""
Domain Library Package - Domain Models and Business Logic

This package provides domain-specific models and business logic including:
- Domain entity definitions
- Business rule templates
- Domain service patterns
- Entity relationship management
- Domain event handling
- Business process modeling
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

# Core imports
from .core import (
    DomainEntity,
    BusinessRule,
    DomainService,
    DomainEvent,
    ValueObject,
)

# Domain imports
from .domain import (
    EntityManager,
    BusinessRuleEngine,
    DomainEventBus,
    ProcessManager,
)

# Templates
from .templates import (
    EntityTemplate,
    ServiceTemplate,
    EventTemplate,
    RuleTemplate,
)

__all__ = [
    # Core
    "DomainEntity",
    "BusinessRule",
    "DomainService",
    "DomainEvent",
    "ValueObject",
    
    # Domain
    "EntityManager",
    "BusinessRuleEngine",
    "DomainEventBus",
    "ProcessManager",
    
    # Templates
    "EntityTemplate",
    "ServiceTemplate",
    "EventTemplate",
    "RuleTemplate",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]