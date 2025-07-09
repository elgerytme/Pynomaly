"""Domain abstractions for core architectural patterns."""

from .base_entity import BaseEntity
from .base_repository import BaseRepository  
from .base_service import BaseService
from .base_value_object import BaseValueObject
from .specification import Specification

__all__ = [
    "BaseEntity",
    "BaseRepository",
    "BaseService", 
    "BaseValueObject",
    "Specification",
]
