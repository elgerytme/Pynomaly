"""User repository interface."""

from abc import abstractmethod
from typing import List, Optional, Protocol
from uuid import UUID

from ..entities.user import User
from ..value_objects.email import Email


class UserRepository(Protocol):
    """User repository interface."""
    
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save user to storage."""
        ...
    
    @abstractmethod
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        """Find user by ID."""
        ...
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        ...
    
    @abstractmethod
    async def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        ...
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Find all users with pagination."""
        ...
    
    @abstractmethod
    async def exists(self, user_id: UUID) -> bool:
        """Check if user exists."""
        ...
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete user."""
        ...
    
    @abstractmethod
    async def count(self) -> int:
        """Count total users."""
        ...