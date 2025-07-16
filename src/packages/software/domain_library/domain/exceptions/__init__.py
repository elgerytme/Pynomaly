"""Domain exceptions for the Domain Library package."""

from .entity_exceptions import (
    InvalidEntityError,
    EntityNotFoundError,
    EntityConflictError
)

from .catalog_exceptions import (
    CatalogError,
    CatalogNotFoundError,
    CatalogAccessError
)

__all__ = [
    "InvalidEntityError",
    "EntityNotFoundError", 
    "EntityConflictError",
    "CatalogError",
    "CatalogNotFoundError",
    "CatalogAccessError"
]