"""Base domain model class for common functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class DomainModel:
    """Base class for domain models."""

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, UUID):
                result[key] = str(value)
            else:
                result[key] = value
        return result
