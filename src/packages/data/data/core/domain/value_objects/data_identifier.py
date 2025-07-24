"""
DataIdentifier

Value object for unique data identifiers.
"""

from dataclasses import dataclass
from software.core.domain.abstractions.base_value_object import BaseValueObject

@dataclass(frozen=True)
class DataIdentifier(BaseValueObject):
    """
    Value object for unique data identifiers.
    """
    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("DataIdentifier value cannot be empty.")

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"DataIdentifier(value='{self.value}')"
