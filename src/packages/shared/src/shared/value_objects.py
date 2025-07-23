"""
Common value objects for the monorepo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Identifier:
    """
    Strongly-typed identifier for domain entities.
    
    Provides type safety and prevents mixing different types of IDs.
    """
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate the identifier."""
        if not self.value:
            raise ValueError("Identifier cannot be empty")
        if not isinstance(self.value, str):
            raise ValueError("Identifier must be a string")
    
    @classmethod
    def generate(cls, prefix: str = "") -> Identifier:
        """Generate a new UUID-based identifier."""
        uid = str(uuid4())
        if prefix:
            return cls(f"{prefix}_{uid}")
        return cls(uid)
    
    @classmethod
    def from_uuid(cls, uid: UUID, prefix: str = "") -> Identifier:
        """Create identifier from UUID."""
        uid_str = str(uid)
        if prefix:
            return cls(f"{prefix}_{uid_str}")
        return cls(uid_str)
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"Identifier('{self.value}')"


@dataclass(frozen=True)
class Email:
    """
    Email address value object with validation.
    """
    
    value: str
    
    # Simple email regex - not RFC compliant but good enough for most use cases
    _EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def __post_init__(self) -> None:
        """Validate the email address."""
        if not self.value:
            raise ValueError("Email cannot be empty")
        if not self._EMAIL_PATTERN.match(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
    
    @property
    def local_part(self) -> str:
        """Get the local part (before @) of the email."""
        return self.value.split('@')[0]
    
    @property
    def domain(self) -> str:
        """Get the domain part (after @) of the email."""
        return self.value.split('@')[1]
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"Email('{self.value}')"


@dataclass(frozen=True)
class Timestamp:
    """
    Consistent timestamp handling across domains.
    """
    
    value: datetime
    
    def __post_init__(self) -> None:
        """Ensure timestamp is timezone-aware."""
        if self.value.tzinfo is None:
            # Convert naive datetime to UTC
            object.__setattr__(self, 'value', self.value.replace(tzinfo=timezone.utc))
    
    @classmethod
    def now(cls) -> Timestamp:
        """Create timestamp for current time."""
        return cls(datetime.now(timezone.utc))
    
    @classmethod
    def from_iso(cls, iso_string: str) -> Timestamp:
        """Create timestamp from ISO format string."""
        dt = datetime.fromisoformat(iso_string)
        return cls(dt)
    
    @classmethod
    def from_unix(cls, unix_timestamp: float) -> Timestamp:
        """Create timestamp from Unix timestamp."""
        dt = datetime.fromtimestamp(unix_timestamp, timezone.utc)
        return cls(dt)
    
    def to_iso(self) -> str:
        """Convert to ISO format string."""
        return self.value.isoformat()
    
    def to_unix(self) -> float:
        """Convert to Unix timestamp."""
        return self.value.timestamp()
    
    def is_before(self, other: Timestamp) -> bool:
        """Check if this timestamp is before another."""
        return self.value < other.value
    
    def is_after(self, other: Timestamp) -> bool:
        """Check if this timestamp is after another."""
        return self.value > other.value
    
    def __str__(self) -> str:
        return self.to_iso()
    
    def __repr__(self) -> str:
        return f"Timestamp('{self.to_iso()}')"


@dataclass(frozen=True)
class Money:
    """
    Financial value representation with currency support.
    """
    
    amount: Decimal
    currency: str
    
    def __post_init__(self) -> None:
        """Validate the money value."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        if not self.currency:
            raise ValueError("Currency cannot be empty")
        if len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter ISO code")
        # Convert to uppercase for consistency
        object.__setattr__(self, 'currency', self.currency.upper())
    
    @classmethod
    def from_float(cls, amount: float, currency: str) -> Money:
        """Create Money from float amount."""
        return cls(Decimal(str(amount)), currency)
    
    @classmethod
    def zero(cls, currency: str) -> Money:
        """Create zero amount in specified currency."""
        return cls(Decimal('0'), currency)
    
    def add(self, other: Money) -> Money:
        """Add two money amounts."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: Money) -> Money:
        """Subtract two money amounts."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(self.amount - other.amount, self.currency)
    
    def multiply(self, factor: Decimal | float | int) -> Money:
        """Multiply money by a factor."""
        if not isinstance(factor, Decimal):
            factor = Decimal(str(factor))
        return Money(self.amount * factor, self.currency)
    
    def divide(self, divisor: Decimal | float | int) -> Money:
        """Divide money by a divisor."""
        if not isinstance(divisor, Decimal):
            divisor = Decimal(str(divisor))
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return Money(self.amount / divisor, self.currency)
    
    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > 0
    
    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < 0
    
    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == 0
    
    def to_float(self) -> float:
        """Convert amount to float (use with caution for precision)."""
        return float(self.amount)
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"
    
    def __repr__(self) -> str:
        return f"Money({self.amount}, '{self.currency}')"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount < other.amount
    
    def __le__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount <= other.amount
    
    def __gt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount > other.amount
    
    def __ge__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount >= other.amount