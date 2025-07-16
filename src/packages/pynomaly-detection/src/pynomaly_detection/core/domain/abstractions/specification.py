"""Specification pattern abstraction."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Specification(Generic[T], ABC):
    """Base specification interface."""

    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """Check if an entity satisfies this specification."""
        pass

    def and_(self, other: "Specification[T]") -> "Specification[T]":
        """Create an AND specification."""
        return AndSpecification(self, other)

    def or_(self, other: "Specification[T]") -> "Specification[T]":
        """Create an OR specification."""
        return OrSpecification(self, other)

    def not_(self) -> "Specification[T]":
        """Create a NOT specification."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies both specifications."""
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)


class OrSpecification(Specification[T]):
    """OR combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies either specification."""
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)


class NotSpecification(Specification[T]):
    """NOT negation of a specification."""

    def __init__(self, specification: Specification[T]) -> None:
        self.specification = specification

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity does not satisfy the specification."""
        return not self.specification.is_satisfied_by(entity)
