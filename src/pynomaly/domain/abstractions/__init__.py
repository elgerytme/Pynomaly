"""Domain abstractions for core architectural patterns."""

from .base_entity import BaseEntity
from .base_repository import BaseRepository
from .base_service import BaseService
from .base_value_object import BaseValueObject
from .domain_event import DomainEvent, DomainEventHandler
from .specification import Specification

__all__ = [
    "BaseEntity",
    "BaseRepository",
    "BaseService",
    "BaseValueObject",
    "DomainEvent",
    "DomainEventHandler",
    "Specification",
]
