"""
Entity-specific exceptions for the Domain Library package.

These exceptions handle error conditions related to domain entity operations.
"""

from typing import Any, Optional


class EntityError(Exception):
    """Base exception for all entity-related errors."""
    
    def __init__(self, message: str, entity_id: Optional[str] = None, **context: Any) -> None:
        super().__init__(message)
        self.entity_id = entity_id
        self.context = context


class InvalidEntityError(EntityError):
    """Raised when entity data is invalid or violates business rules."""
    
    def __init__(self, message: str, field: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.field = field


class EntityNotFoundError(EntityError):
    """Raised when a requested entity cannot be found."""
    
    def __init__(self, entity_id: str, **context: Any) -> None:
        message = f"Entity with ID '{entity_id}' not found"
        super().__init__(message, entity_id=entity_id, **context)


class EntityConflictError(EntityError):
    """Raised when an entity operation conflicts with existing state."""
    
    def __init__(self, message: str, entity_id: Optional[str] = None, conflict_type: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, entity_id=entity_id, **context)
        self.conflict_type = conflict_type


class EntityValidationError(InvalidEntityError):
    """Raised when entity validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[dict] = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.validation_errors = validation_errors or {}


class EntityVersionError(EntityError):
    """Raised when entity versioning operations fail."""
    
    def __init__(self, message: str, entity_id: Optional[str] = None, current_version: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, entity_id=entity_id, **context)
        self.current_version = current_version