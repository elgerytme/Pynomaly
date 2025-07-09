"""Specification pattern for domain queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Specification(Generic[T], ABC):
    """Base specification for domain queries.
    
    The specification pattern allows for flexible, composable query logic
    that can be reused across different contexts.
    """

    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """Check if an entity satisfies this specification.
        
        Args:
            entity: The entity to check
            
        Returns:
            True if the entity satisfies the specification
        """
        ...

    def and_(self, other: Specification[T]) -> Specification[T]:
        """Create an AND specification.
        
        Args:
            other: Another specification
            
        Returns:
            Combined specification with AND logic
        """
        return AndSpecification(self, other)

    def or_(self, other: Specification[T]) -> Specification[T]:
        """Create an OR specification.
        
        Args:
            other: Another specification
            
        Returns:
            Combined specification with OR logic
        """
        return OrSpecification(self, other)

    def not_(self) -> Specification[T]:
        """Create a NOT specification.
        
        Returns:
            Negated specification
        """
        return NotSpecification(self)

    def __and__(self, other: Specification[T]) -> Specification[T]:
        """Support for & operator."""
        return self.and_(other)

    def __or__(self, other: Specification[T]) -> Specification[T]:
        """Support for | operator."""
        return self.or_(other)

    def __invert__(self) -> Specification[T]:
        """Support for ~ operator."""
        return self.not_()

    def to_sql_where(self) -> str:
        """Convert specification to SQL WHERE clause.
        
        Override in subclasses for SQL repositories.
        
        Returns:
            SQL WHERE clause
        """
        raise NotImplementedError("Subclasses must implement to_sql_where")

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary representation.
        
        Returns:
            Dictionary representation of the specification
        """
        return {
            "type": self.__class__.__name__,
            "specification": str(self),
        }


class AndSpecification(Specification[T]):
    """AND combination of two specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        """Initialize AND specification.
        
        Args:
            left: Left specification
            right: Right specification
        """
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies both specifications."""
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        return f"({self.left.to_sql_where()}) AND ({self.right.to_sql_where()})"

    def __str__(self) -> str:
        """String representation."""
        return f"({self.left} AND {self.right})"


class OrSpecification(Specification[T]):
    """OR combination of two specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        """Initialize OR specification.
        
        Args:
            left: Left specification
            right: Right specification
        """
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies either specification."""
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        return f"({self.left.to_sql_where()}) OR ({self.right.to_sql_where()})"

    def __str__(self) -> str:
        """String representation."""
        return f"({self.left} OR {self.right})"


class NotSpecification(Specification[T]):
    """NOT negation of a specification."""

    def __init__(self, specification: Specification[T]) -> None:
        """Initialize NOT specification.
        
        Args:
            specification: Specification to negate
        """
        self.specification = specification

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity does not satisfy the specification."""
        return not self.specification.is_satisfied_by(entity)

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        return f"NOT ({self.specification.to_sql_where()})"

    def __str__(self) -> str:
        """String representation."""
        return f"NOT ({self.specification})"


class FieldSpecification(Specification[T]):
    """Specification for field-based queries."""

    def __init__(self, field_name: str, operator: str, value: Any) -> None:
        """Initialize field specification.
        
        Args:
            field_name: Name of the field
            operator: Comparison operator (eq, ne, lt, le, gt, ge, in, not_in, like, etc.)
            value: Value to compare against
        """
        self.field_name = field_name
        self.operator = operator
        self.value = value

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity field satisfies the condition."""
        field_value = getattr(entity, self.field_name, None)
        
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "ne":
            return field_value != self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "le":
            return field_value <= self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "ge":
            return field_value >= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "like":
            return str(self.value).lower() in str(field_value).lower()
        elif self.operator == "is_null":
            return field_value is None
        elif self.operator == "is_not_null":
            return field_value is not None
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        if self.operator == "eq":
            return f"{self.field_name} = %s"
        elif self.operator == "ne":
            return f"{self.field_name} != %s"
        elif self.operator == "lt":
            return f"{self.field_name} < %s"
        elif self.operator == "le":
            return f"{self.field_name} <= %s"
        elif self.operator == "gt":
            return f"{self.field_name} > %s"
        elif self.operator == "ge":
            return f"{self.field_name} >= %s"
        elif self.operator == "in":
            placeholders = ", ".join(["%s"] * len(self.value))
            return f"{self.field_name} IN ({placeholders})"
        elif self.operator == "not_in":
            placeholders = ", ".join(["%s"] * len(self.value))
            return f"{self.field_name} NOT IN ({placeholders})"
        elif self.operator == "like":
            return f"{self.field_name} LIKE %s"
        elif self.operator == "is_null":
            return f"{self.field_name} IS NULL"
        elif self.operator == "is_not_null":
            return f"{self.field_name} IS NOT NULL"
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def __str__(self) -> str:
        """String representation."""
        return f"{self.field_name} {self.operator} {self.value}"


class CompositeSpecification(Specification[T]):
    """Composite specification for complex queries."""

    def __init__(self, specifications: list[Specification[T]], operator: str = "and") -> None:
        """Initialize composite specification.
        
        Args:
            specifications: List of specifications to combine
            operator: Logical operator ("and" or "or")
        """
        self.specifications = specifications
        self.operator = operator.lower()

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies the composite specification."""
        if not self.specifications:
            return True
        
        if self.operator == "and":
            return all(spec.is_satisfied_by(entity) for spec in self.specifications)
        elif self.operator == "or":
            return any(spec.is_satisfied_by(entity) for spec in self.specifications)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        if not self.specifications:
            return "1=1"
        
        clauses = [spec.to_sql_where() for spec in self.specifications]
        separator = " AND " if self.operator == "and" else " OR "
        return "(" + separator.join(clauses) + ")"

    def __str__(self) -> str:
        """String representation."""
        separator = " AND " if self.operator == "and" else " OR "
        return "(" + separator.join(str(spec) for spec in self.specifications) + ")"


class AlwaysTrueSpecification(Specification[T]):
    """Specification that always returns true."""

    def is_satisfied_by(self, entity: T) -> bool:
        """Always returns true."""
        return True

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        return "1=1"

    def __str__(self) -> str:
        """String representation."""
        return "TRUE"


class AlwaysFalseSpecification(Specification[T]):
    """Specification that always returns false."""

    def is_satisfied_by(self, entity: T) -> bool:
        """Always returns false."""
        return False

    def to_sql_where(self) -> str:
        """Convert to SQL WHERE clause."""
        return "1=0"

    def __str__(self) -> str:
        """String representation."""
        return "FALSE"


# Convenience functions for creating specifications
def field_equals(field_name: str, value: Any) -> FieldSpecification:
    """Create a field equals specification."""
    return FieldSpecification(field_name, "eq", value)


def field_not_equals(field_name: str, value: Any) -> FieldSpecification:
    """Create a field not equals specification."""
    return FieldSpecification(field_name, "ne", value)


def field_less_than(field_name: str, value: Any) -> FieldSpecification:
    """Create a field less than specification."""
    return FieldSpecification(field_name, "lt", value)


def field_less_than_or_equal(field_name: str, value: Any) -> FieldSpecification:
    """Create a field less than or equal specification."""
    return FieldSpecification(field_name, "le", value)


def field_greater_than(field_name: str, value: Any) -> FieldSpecification:
    """Create a field greater than specification."""
    return FieldSpecification(field_name, "gt", value)


def field_greater_than_or_equal(field_name: str, value: Any) -> FieldSpecification:
    """Create a field greater than or equal specification."""
    return FieldSpecification(field_name, "ge", value)


def field_in(field_name: str, values: list[Any]) -> FieldSpecification:
    """Create a field in specification."""
    return FieldSpecification(field_name, "in", values)


def field_not_in(field_name: str, values: list[Any]) -> FieldSpecification:
    """Create a field not in specification."""
    return FieldSpecification(field_name, "not_in", values)


def field_like(field_name: str, pattern: str) -> FieldSpecification:
    """Create a field like specification."""
    return FieldSpecification(field_name, "like", pattern)


def field_is_null(field_name: str) -> FieldSpecification:
    """Create a field is null specification."""
    return FieldSpecification(field_name, "is_null", None)


def field_is_not_null(field_name: str) -> FieldSpecification:
    """Create a field is not null specification."""
    return FieldSpecification(field_name, "is_not_null", None)


def always_true() -> AlwaysTrueSpecification:
    """Create an always true specification."""
    return AlwaysTrueSpecification()


def always_false() -> AlwaysFalseSpecification:
    """Create an always false specification."""
    return AlwaysFalseSpecification()


def combine_and(*specifications: Specification[T]) -> CompositeSpecification[T]:
    """Combine specifications with AND logic."""
    return CompositeSpecification(list(specifications), "and")


def combine_or(*specifications: Specification[T]) -> CompositeSpecification[T]:
    """Combine specifications with OR logic."""
    return CompositeSpecification(list(specifications), "or")
