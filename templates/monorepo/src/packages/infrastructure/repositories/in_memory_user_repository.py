"""In-memory user repository implementation."""

from typing import Dict, List, Optional
from uuid import UUID

from packages.core.domain.entities.user import User
from packages.core.domain.repositories.user_repository import UserRepository
from packages.core.domain.value_objects.email import Email


class InMemoryUserRepository(UserRepository):
    """In-memory implementation of user repository."""
    
    def __init__(self) -> None:
        self._users: Dict[UUID, User] = {}
    
    async def save(self, user: User) -> User:
        """Save user to memory."""
        self._users[user.id] = user
        return user
    
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        """Find user by ID."""
        return self._users.get(user_id)
    
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        for user in self._users.values():
            if user.email == email.value:
                return user
        return None
    
    async def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Find all users with pagination."""
        users = list(self._users.values())
        return users[offset:offset + limit]
    
    async def exists(self, user_id: UUID) -> bool:
        """Check if user exists."""
        return user_id in self._users
    
    async def delete(self, user_id: UUID) -> bool:
        """Delete user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
    
    async def count(self) -> int:
        """Count total users."""
        return len(self._users)