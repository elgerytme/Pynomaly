"""Create user use case."""

from typing import Protocol

from ..dto.user_dto import CreateUserDTO, UserDTO
from ..entities.user import User
from ..repositories.user_repository import UserRepository
from ..value_objects.email import Email


class PasswordHasher(Protocol):
    """Password hashing interface."""
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        ...


class CreateUserUseCase:
    """Use case for creating a new user."""
    
    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
    ) -> None:
        self._user_repository = user_repository
        self._password_hasher = password_hasher
    
    async def execute(self, dto: CreateUserDTO) -> UserDTO:
        """Create a new user."""
        # Check if email already exists
        email = Email(value=dto.email)
        existing_user = await self._user_repository.find_by_email(email)
        if existing_user:
            raise ValueError(f"User with email {dto.email} already exists")
        
        # Check if username already exists
        existing_user = await self._user_repository.find_by_username(dto.username)
        if existing_user:
            raise ValueError(f"User with username {dto.username} already exists")
        
        # Hash password
        hashed_password = self._password_hasher.hash_password(dto.password)
        
        # Create user entity
        user = User(
            email=dto.email,
            username=dto.username,
            full_name=dto.full_name,
            hashed_password=hashed_password,
            roles=dto.roles,
        )
        
        # Save user
        saved_user = await self._user_repository.save(user)
        
        # Return DTO
        return UserDTO(
            id=str(saved_user.id),
            email=saved_user.email,
            username=saved_user.username,
            full_name=saved_user.full_name,
            roles=saved_user.roles,
            status=saved_user.status,
            created_at=saved_user.created_at,
            updated_at=saved_user.updated_at,
            last_login=saved_user.last_login,
            email_verified=saved_user.email_verified,
            failed_login_attempts=saved_user.failed_login_attempts,
            is_locked=saved_user.is_locked(),
        )