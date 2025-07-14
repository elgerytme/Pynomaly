"""User management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from packages.core.application.dto.user_dto import CreateUserDTO, UpdateUserDTO, UserDTO
from packages.core.application.use_cases.create_user import CreateUserUseCase
from packages.core.domain.repositories.user_repository import UserRepository
from packages.infrastructure.repositories.in_memory_user_repository import InMemoryUserRepository
from packages.infrastructure.security.password_hasher import BCryptPasswordHasher

router = APIRouter()


def get_user_repository() -> UserRepository:
    """Get user repository dependency."""
    return InMemoryUserRepository()


def get_password_hasher() -> BCryptPasswordHasher:
    """Get password hasher dependency."""
    return BCryptPasswordHasher()


def get_create_user_use_case(
    user_repository: UserRepository = Depends(get_user_repository),
    password_hasher: BCryptPasswordHasher = Depends(get_password_hasher),
) -> CreateUserUseCase:
    """Get create user use case dependency."""
    return CreateUserUseCase(user_repository, password_hasher)


@router.post("/", response_model=UserDTO, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: CreateUserDTO,
    use_case: CreateUserUseCase = Depends(get_create_user_use_case),
) -> UserDTO:
    """Create a new user."""
    try:
        return await use_case.execute(user_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{user_id}", response_model=UserDTO)
async def get_user(
    user_id: UUID,
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserDTO:
    """Get user by ID."""
    user = await user_repository.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserDTO(
        id=str(user.id),
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        roles=user.roles,
        status=user.status,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
        email_verified=user.email_verified,
        failed_login_attempts=user.failed_login_attempts,
        is_locked=user.is_locked(),
    )


@router.get("/", response_model=List[UserDTO])
async def list_users(
    limit: int = 100,
    offset: int = 0,
    user_repository: UserRepository = Depends(get_user_repository),
) -> List[UserDTO]:
    """List all users."""
    users = await user_repository.find_all(limit=limit, offset=offset)
    
    return [
        UserDTO(
            id=str(user.id),
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            roles=user.roles,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login,
            email_verified=user.email_verified,
            failed_login_attempts=user.failed_login_attempts,
            is_locked=user.is_locked(),
        )
        for user in users
    ]


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    user_repository: UserRepository = Depends(get_user_repository),
) -> None:
    """Delete user."""
    if not await user_repository.exists(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    await user_repository.delete(user_id)