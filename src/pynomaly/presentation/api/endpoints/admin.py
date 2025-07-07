"""Admin management endpoints for RBAC."""


from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr

from pynomaly.infrastructure.auth import (
    get_auth,
    require_admin,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container, get_current_user

router = APIRouter()


class CreateUserRequest(BaseModel):
    """Request model for creating a user."""

    username: str
    email: EmailStr
    password: str
    full_name: str | None = None
    roles: list[str] = ["user"]
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""

    username: str | None = None
    email: EmailStr | None = None
    full_name: str | None = None
    roles: list[str] | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None


class UserResponse(BaseModel):
    """Response model for user data."""

    id: str
    username: str
    email: str
    full_name: str | None
    roles: list[str]
    is_active: bool
    is_superuser: bool
    created_at: str
    last_login: str | None
    api_keys: list[str]


class CreateApiKeyRequest(BaseModel):
    """Request model for creating an API key."""

    key_name: str
    user_id: str


class ApiKeyResponse(BaseModel):
    """Response model for API key creation."""

    api_key: str
    user_id: str
    key_name: str


class RoleInfo(BaseModel):
    """Role information with permissions."""

    role: str
    permissions: list[str]
    description: str


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    is_active: bool | None = Query(None, description="Filter by active status"),
    role: str | None = Query(None, description="Filter by role"),
    limit: int = Query(100, ge=1, le=1000),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> list[UserResponse]:
    """List all users. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Get all users from auth service
    users = list(auth_service._users.values())

    # Apply filters
    if is_active is not None:
        users = [u for u in users if u.is_active == is_active]

    if role:
        users = [u for u in users if role in u.roles]

    # Limit results
    users = users[:limit]

    # Convert to response models
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            roles=u.roles,
            is_active=u.is_active,
            is_superuser=u.is_superuser,
            created_at=u.created_at.isoformat(),
            last_login=u.last_login.isoformat() if u.last_login else None,
            api_keys=u.api_keys,
        )
        for u in users
    ]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> UserResponse:
    """Get a specific user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=user.roles,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
        api_keys=user.api_keys,
    )


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: CreateUserRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> UserResponse:
    """Create a new user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    try:
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            roles=user_data.roles,
        )

        # Update active status if different from default
        if not user_data.is_active:
            user.is_active = user_data.is_active

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=user.roles,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None,
            api_keys=user.api_keys,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UpdateUserRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> UserResponse:
    """Update a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update fields
    if update_data.username is not None:
        # Check if username is already taken
        for existing_user in auth_service._users.values():
            if (
                existing_user.id != user_id
                and existing_user.username == update_data.username
            ):
                raise HTTPException(status_code=400, detail="Username already taken")
        user.username = update_data.username

    if update_data.email is not None:
        # Check if email is already taken
        for existing_user in auth_service._users.values():
            if existing_user.id != user_id and existing_user.email == update_data.email:
                raise HTTPException(status_code=400, detail="Email already taken")
        user.email = update_data.email

    if update_data.full_name is not None:
        user.full_name = update_data.full_name

    if update_data.roles is not None:
        user.roles = update_data.roles

    if update_data.is_active is not None:
        user.is_active = update_data.is_active

    if update_data.is_superuser is not None:
        user.is_superuser = update_data.is_superuser

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=user.roles,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
        api_keys=user.api_keys,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> dict:
    """Delete a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Prevent self-deletion
    current_user_obj = None
    if current_user:
        for user in auth_service._users.values():
            if user.username == current_user:
                current_user_obj = user
                break

    if current_user_obj and current_user_obj.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    if user_id not in auth_service._users:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove user and their API keys
    user = auth_service._users.pop(user_id)
    for api_key in user.api_keys:
        auth_service._api_keys.pop(api_key, None)

    return {"success": True, "message": f"User {user.username} deleted"}


@router.post("/users/{user_id}/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    user_id: str,
    api_key_data: CreateApiKeyRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> ApiKeyResponse:
    """Create an API key for a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    try:
        api_key = auth_service.create_api_key(user_id, api_key_data.key_name)
        return ApiKeyResponse(
            api_key=api_key, user_id=user_id, key_name=api_key_data.key_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> dict:
    """Revoke an API key. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    success = auth_service.revoke_api_key(api_key)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"success": True, "message": "API key revoked"}


@router.get("/roles", response_model=list[RoleInfo])
async def list_roles(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> list[RoleInfo]:
    """List all available roles and their permissions. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Get role-permission mapping
    role_descriptions = {
        "admin": "Full system access including user management and system configuration",
        "user": "Standard user with read/write access to detectors, datasets, and experiments",
        "viewer": "Read-only access to view detectors, datasets, and experiments",
    }

    roles = []
    for role in ["admin", "user", "viewer"]:
        permissions = auth_service._get_permissions_for_roles([role])
        roles.append(
            RoleInfo(
                role=role,
                permissions=permissions,
                description=role_descriptions.get(role, ""),
            )
        )

    return roles


@router.get("/permissions")
async def list_permissions(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> dict:
    """List all available permissions organized by resource. Requires admin permissions."""
    return {
        "detectors": ["detectors:read", "detectors:write", "detectors:delete"],
        "datasets": ["datasets:read", "datasets:write", "datasets:delete"],
        "experiments": ["experiments:read", "experiments:write", "experiments:delete"],
        "users": ["users:read", "users:write", "users:delete"],
        "settings": ["settings:read", "settings:write"],
    }


@router.get("/users/{user_id}/permissions")
async def get_user_permissions(
    user_id: str,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_admin),
) -> dict:
    """Get effective permissions for a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    permissions = auth_service._get_permissions_for_roles(user.roles)

    return {
        "user_id": user_id,
        "username": user.username,
        "roles": user.roles,
        "permissions": permissions,
        "is_superuser": user.is_superuser,
    }
