"""Base repository abstraction - DEPRECATED.

This module is deprecated. Use `pynomaly.shared.protocols.repository_protocol` instead.
All new repositories should implement the async RepositoryProtocol interface.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID
from warnings import warn

T = TypeVar("T")


class BaseRepository(Generic[T], ABC):
    """Base repository interface.

    DEPRECATED: Use RepositoryProtocol from pynomaly_detection.shared.protocols instead.
    This class exists only for backward compatibility and will be removed in v2.0.
    """

    def __init__(self):
        """Initialize repository with deprecation warning."""
        warn(
            "BaseRepository is deprecated. Use RepositoryProtocol from "
            "pynomaly.shared.protocols.repository_protocol instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save an entity."""
        pass

    @abstractmethod
    async def find_by_id(self, entity_id: UUID) -> T | None:
        """Find entity by ID."""
        pass

    @abstractmethod
    async def find_all(self) -> list[T]:
        """Find all entities."""
        pass

    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Count total entities."""
        pass
