"""
Shared package for common utilities and types across the monorepo.

This package provides foundational components that are used across multiple 
domains while maintaining domain independence and following DDD principles.
"""

__version__ = "0.1.0"

from .types import Result, Success, Failure, Optional, Paginated, ValidationResult
from .value_objects import Identifier, Email, Timestamp, Money
from .base_classes import Entity, ValueObject, DomainEvent, UseCase

__all__ = [
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
]