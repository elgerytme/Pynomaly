"""Base value object for immutable domain values."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseValueObject")


class BaseValueObject(BaseModel, ABC):
    """Base class for value objects in the domain.
    
    Value objects are immutable objects that represent a value
    and are distinguished by their attributes rather than identity.
    """

    # Class-level configuration
    __value_object_type__: ClassVar[str] = ""

    class Config:
        """Pydantic configuration for value objects."""
        # Value objects are immutable
        allow_mutation = False
        # Validate on assignment
        validate_assignment = True
        # Use enum values for serialization
        use_enum_values = True
        # Custom validation
        validate_all = True
        # Arbitrary types allowed
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        """Initialize value object with validation."""
        super().__init__(**data)
        
        # Set value object type if not already set
        if not self.__value_object_type__:
            self.__value_object_type__ = self.__class__.__name__

    def __hash__(self) -> int:
        """Hash based on all field values."""
        return hash(tuple(sorted(self.dict().items())))

    def __eq__(self, other: object) -> bool:
        """Equality based on all field values."""
        if not isinstance(other, self.__class__):
            return False
        return self.dict() == other.dict()

    def __str__(self) -> str:
        """String representation showing all field values."""
        fields = ", ".join(f"{k}={v}" for k, v in self.dict().items())
        return f"{self.__class__.__name__}({fields})"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__str__()

    def to_dict(self) -> dict[str, Any]:
        """Convert value object to dictionary.
        
        Returns:
            Dictionary representation of the value object
        """
        return self.dict()

    def to_json(self) -> str:
        """Convert value object to JSON string.
        
        Returns:
            JSON string representation
        """
        return self.json()

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create value object from dictionary.
        
        Args:
            data: Dictionary with value object data
            
        Returns:
            New value object instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Create value object from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New value object instance
        """
        return cls.parse_raw(json_str)

    def validate_invariants(self) -> None:
        """Validate value object invariants.
        
        Override in subclasses to add domain-specific validation.
        
        Raises:
            ValueError: If invariants are violated
        """
        # Base validation is handled by Pydantic
        pass

    def with_updates(self: T, **updates: Any) -> T:
        """Create a new value object with updated values.
        
        Since value objects are immutable, this creates a new instance
        with the specified updates.
        
        Args:
            **updates: Field updates
            
        Returns:
            New value object instance with updates
        """
        current_data = self.dict()
        current_data.update(updates)
        return self.__class__(**current_data)

    def is_valid(self) -> bool:
        """Check if value object is valid.
        
        Returns:
            True if value object is valid
        """
        try:
            self.validate_invariants()
            return True
        except ValueError:
            return False

    def get_value_object_type(self) -> str:
        """Get the value object type.
        
        Returns:
            Value object type name
        """
        return self.__value_object_type__

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Get the JSON schema for this value object.
        
        Returns:
            JSON schema dictionary
        """
        return cls.schema()

    @classmethod
    def get_field_names(cls) -> list[str]:
        """Get the field names for this value object.
        
        Returns:
            List of field names
        """
        return list(cls.__fields__.keys())

    @classmethod
    def get_field_types(cls) -> dict[str, type]:
        """Get the field types for this value object.
        
        Returns:
            Dictionary mapping field names to types
        """
        return {name: field.type_ for name, field in cls.__fields__.items()}

    def compare_to(self, other: BaseValueObject) -> int:
        """Compare this value object to another.
        
        Args:
            other: Another value object
            
        Returns:
            -1 if this < other, 0 if equal, 1 if this > other
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot compare {self.__class__} with {other.__class__}")
        
        self_dict = self.dict()
        other_dict = other.dict()
        
        if self_dict < other_dict:
            return -1
        elif self_dict > other_dict:
            return 1
        else:
            return 0

    def __lt__(self, other: BaseValueObject) -> bool:
        """Less than comparison."""
        return self.compare_to(other) < 0

    def __le__(self, other: BaseValueObject) -> bool:
        """Less than or equal comparison."""
        return self.compare_to(other) <= 0

    def __gt__(self, other: BaseValueObject) -> bool:
        """Greater than comparison."""
        return self.compare_to(other) > 0

    def __ge__(self, other: BaseValueObject) -> bool:
        """Greater than or equal comparison."""
        return self.compare_to(other) >= 0


class SingleValueObject(BaseValueObject, Generic[T]):
    """Base class for single-value objects."""

    value: T

    def __init__(self, value: T) -> None:
        """Initialize single value object.
        
        Args:
            value: The value to wrap
        """
        super().__init__(value=value)

    def __str__(self) -> str:
        """String representation showing only the value."""
        return str(self.value)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}({self.value!r})"

    def get_value(self) -> T:
        """Get the wrapped value.
        
        Returns:
            The wrapped value
        """
        return self.value

    def __hash__(self) -> int:
        """Hash based on the value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Equality based on the value."""
        if isinstance(other, SingleValueObject):
            return self.value == other.value
        return self.value == other

    def compare_to(self, other: SingleValueObject[T]) -> int:
        """Compare based on the wrapped value."""
        if self.value < other.value:
            return -1
        elif self.value > other.value:
            return 1
        else:
            return 0


class CompositeValueObject(BaseValueObject):
    """Base class for composite value objects with multiple fields."""

    def get_components(self) -> dict[str, Any]:
        """Get all components of the composite value object.
        
        Returns:
            Dictionary of all field values
        """
        return self.dict()

    def get_component(self, name: str) -> Any:
        """Get a specific component value.
        
        Args:
            name: Name of the component
            
        Returns:
            Component value
        """
        return getattr(self, name)

    def has_component(self, name: str) -> bool:
        """Check if a component exists.
        
        Args:
            name: Name of the component
            
        Returns:
            True if component exists
        """
        return hasattr(self, name)

    def validate_components(self) -> None:
        """Validate all components.
        
        Override in subclasses to add component-specific validation.
        """
        pass

    def validate_invariants(self) -> None:
        """Validate value object invariants including components."""
        super().validate_invariants()
        self.validate_components()


class RangeValueObject(BaseValueObject, Generic[T]):
    """Base class for range value objects."""

    min_value: T
    max_value: T

    def __init__(self, min_value: T, max_value: T) -> None:
        """Initialize range value object.
        
        Args:
            min_value: Minimum value
            max_value: Maximum value
        """
        super().__init__(min_value=min_value, max_value=max_value)

    def contains(self, value: T) -> bool:
        """Check if value is within the range.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is within range
        """
        return self.min_value <= value <= self.max_value

    def overlaps(self, other: RangeValueObject[T]) -> bool:
        """Check if this range overlaps with another.
        
        Args:
            other: Another range value object
            
        Returns:
            True if ranges overlap
        """
        return (self.min_value <= other.max_value and 
                other.min_value <= self.max_value)

    def get_size(self) -> T:
        """Get the size of the range.
        
        Returns:
            Size of the range
        """
        return self.max_value - self.min_value

    def is_empty(self) -> bool:
        """Check if the range is empty.
        
        Returns:
            True if range is empty
        """
        return self.min_value > self.max_value

    def validate_invariants(self) -> None:
        """Validate range invariants."""
        super().validate_invariants()
        if self.min_value > self.max_value:
            raise ValueError("Min value cannot be greater than max value")


class EnumValueObject(BaseValueObject):
    """Base class for enum-like value objects."""

    @classmethod
    def get_valid_values(cls) -> list[Any]:
        """Get list of valid values for this enum.
        
        Override in subclasses to define valid values.
        
        Returns:
            List of valid values
        """
        return []

    def validate_invariants(self) -> None:
        """Validate that value is one of the valid values."""
        super().validate_invariants()
        valid_values = self.get_valid_values()
        if valid_values and self.dict() not in valid_values:
            raise ValueError(f"Invalid value: {self.dict()}. Valid values: {valid_values}")


class MoneyValueObject(BaseValueObject):
    """Base class for money value objects."""

    amount: float
    currency: str

    def __init__(self, amount: float, currency: str) -> None:
        """Initialize money value object.
        
        Args:
            amount: Monetary amount
            currency: Currency code (e.g., 'USD', 'EUR')
        """
        super().__init__(amount=amount, currency=currency)

    def add(self, other: MoneyValueObject) -> MoneyValueObject:
        """Add two money values.
        
        Args:
            other: Another money value object
            
        Returns:
            New money value object with sum
        """
        if self.currency != other.currency:
            raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
        
        return self.__class__(self.amount + other.amount, self.currency)

    def subtract(self, other: MoneyValueObject) -> MoneyValueObject:
        """Subtract two money values.
        
        Args:
            other: Another money value object
            
        Returns:
            New money value object with difference
        """
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract different currencies: {self.currency} and {other.currency}")
        
        return self.__class__(self.amount - other.amount, self.currency)

    def multiply(self, factor: float) -> MoneyValueObject:
        """Multiply money value by a factor.
        
        Args:
            factor: Multiplication factor
            
        Returns:
            New money value object with product
        """
        return self.__class__(self.amount * factor, self.currency)

    def divide(self, divisor: float) -> MoneyValueObject:
        """Divide money value by a divisor.
        
        Args:
            divisor: Division divisor
            
        Returns:
            New money value object with quotient
        """
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        
        return self.__class__(self.amount / divisor, self.currency)

    def is_zero(self) -> bool:
        """Check if amount is zero.
        
        Returns:
            True if amount is zero
        """
        return self.amount == 0

    def is_positive(self) -> bool:
        """Check if amount is positive.
        
        Returns:
            True if amount is positive
        """
        return self.amount > 0

    def is_negative(self) -> bool:
        """Check if amount is negative.
        
        Returns:
            True if amount is negative
        """
        return self.amount < 0

    def validate_invariants(self) -> None:
        """Validate money invariants."""
        super().validate_invariants()
        if not self.currency:
            raise ValueError("Currency cannot be empty")
        if len(self.currency) != 3:
            raise ValueError("Currency must be a 3-character code")
