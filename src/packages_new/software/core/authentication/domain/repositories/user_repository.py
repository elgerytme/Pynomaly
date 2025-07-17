"""
User repository interface for authentication domain
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from ..entities.user import User
from ..value_objects.email import Email

class UserRepository(ABC):
    """Abstract repository interface for User entities"""
    
    @abstractmethod
    def save(self, user: User) -> User:
        """Save a user entity"""
        pass
    
    @abstractmethod
    def find_by_id(self, user_id: UUID) -> Optional[User]:
        """Find user by ID"""
        pass
    
    @abstractmethod
    def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email"""
        pass
    
    @abstractmethod
    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        pass
    
    @abstractmethod
    def find_active_users(self) -> List[User]:
        """Find all active users"""
        pass
    
    @abstractmethod
    def find_unverified_users(self) -> List[User]:
        """Find all unverified users"""
        pass
    
    @abstractmethod
    def delete(self, user_id: UUID) -> bool:
        """Delete a user"""
        pass
    
    @abstractmethod
    def exists_by_email(self, email: Email) -> bool:
        """Check if user exists by email"""
        pass
    
    @abstractmethod
    def exists_by_username(self, username: str) -> bool:
        """Check if user exists by username"""
        pass