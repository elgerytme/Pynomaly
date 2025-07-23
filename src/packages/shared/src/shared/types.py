"""
Common types for type-safe error handling and data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union, List, Dict

T = TypeVar("T")
E = TypeVar("E")


class Result(Generic[T, E], ABC):
    """
    Type-safe result pattern for error handling.
    
    Represents either a successful value (Success) or an error (Failure).
    Prevents the need for exception handling in many cases.
    """
    
    @abstractmethod
    def is_success(self) -> bool:
        """Check if result represents success."""
        pass
    
    @abstractmethod
    def is_failure(self) -> bool:
        """Check if result represents failure."""
        pass
    
    @property
    @abstractmethod
    def value(self) -> T:
        """Get the success value. Raises if called on Failure."""
        pass
    
    @property
    @abstractmethod
    def error(self) -> E:
        """Get the error value. Raises if called on Success."""
        pass
    
    def map(self, func) -> Result[Any, E]:
        """Transform the success value if present."""
        if self.is_success():
            try:
                return Success(func(self.value))
            except Exception as e:
                return Failure(e)
        return self
    
    def flat_map(self, func) -> Result[Any, E]:
        """Chain operations that return Results."""
        if self.is_success():
            return func(self.value)
        return self


@dataclass(frozen=True)
class Success(Result[T, E]):
    """Represents a successful result."""
    
    _value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_failure(self) -> bool:
        return False
    
    @property
    def value(self) -> T:
        return self._value
    
    @property
    def error(self) -> E:
        raise RuntimeError("Cannot get error from Success")


@dataclass(frozen=True)
class Failure(Result[T, E]):
    """Represents a failed result."""
    
    _error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_failure(self) -> bool:
        return True
    
    @property
    def value(self) -> T:
        raise RuntimeError("Cannot get value from Failure")
    
    @property
    def error(self) -> E:
        return self._error


@dataclass(frozen=True)
class Optional(Generic[T]):
    """
    Explicit optional type that forces null checking.
    
    Similar to Rust's Option or Haskell's Maybe.
    """
    
    _value: T | None
    
    @classmethod
    def some(cls, value: T) -> Optional[T]:
        """Create an Optional with a value."""
        if value is None:
            raise ValueError("Cannot create Some with None value")
        return cls(value)
    
    @classmethod
    def none(cls) -> Optional[T]:
        """Create an empty Optional."""
        return cls(None)
    
    def is_some(self) -> bool:
        """Check if Optional contains a value."""
        return self._value is not None
    
    def is_none(self) -> bool:
        """Check if Optional is empty."""
        return self._value is None
    
    def unwrap(self) -> T:
        """Get the value, raising if None."""
        if self._value is None:
            raise RuntimeError("Called unwrap on None Optional")
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        """Get the value or return default."""
        return self._value if self._value is not None else default
    
    def map(self, func) -> Optional[Any]:
        """Transform the value if present."""
        if self._value is not None:
            return Optional.some(func(self._value))
        return Optional.none()


@dataclass(frozen=True)
class Paginated(Generic[T]):
    """
    Standardized pagination response.
    """
    
    items: List[T]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total_items: int,
        page: int,
        page_size: int,
    ) -> Paginated[T]:
        """Create a paginated response with calculated fields."""
        total_pages = (total_items + page_size - 1) // page_size
        has_next = page < total_pages
        has_previous = page > 1
        
        return cls(
            items=items,
            total_items=total_items,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous,
        )


@dataclass(frozen=True)
class ValidationError:
    """Represents a single validation error."""
    
    field: str
    value: Any
    message: str
    code: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of input validation.
    """
    
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str] | None = None
    
    @classmethod
    def success(cls, warnings: List[str] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, errors=[], warnings=warnings)
    
    @classmethod
    def failure(
        cls, 
        errors: List[ValidationError],
        warnings: List[str] | None = None
    ) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings)
    
    def add_error(self, field: str, value: Any, message: str, code: str | None = None) -> ValidationResult:
        """Add an error to the validation result."""
        new_error = ValidationError(field=field, value=value, message=message, code=code)
        return ValidationResult(
            is_valid=False,
            errors=self.errors + [new_error],
            warnings=self.warnings
        )
    
    def get_errors_for_field(self, field: str) -> List[ValidationError]:
        """Get all errors for a specific field."""
        return [error for error in self.errors if error.field == field]
    
    def get_error_messages(self) -> List[str]:
        """Get all error messages as strings."""
        return [error.message for error in self.errors]