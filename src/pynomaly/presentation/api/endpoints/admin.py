"""Comprehensive admin management endpoints for user management and system administration."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from pynomaly.infrastructure.auth import (
    UserModel,
    get_auth,
    require_super_admin,
    require_tenant_admin,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.auth_deps import get_container_simple

router = APIRouter()


class CreateUserRequest(BaseModel):
    """Request model for creating a user."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(None, max_length=100)
    roles: list[str] = Field(default=["user"])
    is_active: bool = True
    is_superuser: bool = False


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""

    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = Field(None, min_length=5, max_length=100)
    full_name: str | None = Field(None, max_length=100)
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
    api_key_count: int
    failed_login_attempts: int = 0
    account_locked: bool = False


class CreateApiKeyRequest(BaseModel):
    """Request model for creating an API key."""

    key_name: str = Field(..., min_length=1, max_length=50)
    user_id: str


class ApiKeyResponse(BaseModel):
    """Response model for API key creation."""

    api_key: str
    user_id: str
    key_name: str
    created_at: str
    expires_at: str | None = None


class RoleInfo(BaseModel):
    """Role information with permissions."""

    role: str
    permissions: list[str]
    description: str
    user_count: int = 0


class PasswordResetRequest(BaseModel):
    """Admin password reset request model."""

    new_password: str = Field(..., min_length=8, max_length=128)
    force_change_on_login: bool = True


class UserListResponse(BaseModel):
    """User list response with pagination."""

    users: list[UserResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class SystemStatsResponse(BaseModel):
    """System statistics response model."""

    total_users: int
    active_users: int
    inactive_users: int
    admin_users: int
    total_api_keys: int
    recent_logins: int
    failed_login_attempts: int
    system_uptime: str


class AuditEvent(BaseModel):
    """Audit event model."""

    event_id: str
    user_id: str
    username: str
    event_type: str
    event_details: dict
    timestamp: str
    ip_address: str | None = None
    user_agent: str | None = None
    admin_user_id: str | None = None


@router.get("/users", response_model=UserListResponse)
async def list_users(
    is_active: bool | None = Query(None, description="Filter by active status"),
    role: str | None = Query(None, description="Filter by role"),
    search: str | None = Query(None, description="Search users by username or email"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    container: Container = Depends(get_container_simple),
    _user: UserModel = Depends(require_super_admin),
) -> UserListResponse:
    """List all users with pagination and filtering. Requires admin permissions."""
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

    if search:
        users = [
            u
            for u in users
            if search.lower() in u.username.lower() or search.lower() in u.email.lower()
        ]

    # Pagination
    total = len(users)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_users = users[start_idx:end_idx]

    # Convert to response models
    user_responses = [
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
            api_key_count=len(u.api_keys),
            failed_login_attempts=len(
                auth_service._failed_login_attempts.get(u.username, [])
            ),
            account_locked=len(auth_service._failed_login_attempts.get(u.username, []))
            >= 5,
        )
        for u in paginated_users
    ]

    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end_idx < total,
        has_previous=page > 1,
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    container: Container = Depends(get_container_simple),
    _user: UserModel = Depends(require_tenant_admin),
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
        api_key_count=len(user.api_keys),
        failed_login_attempts=len(
            auth_service._failed_login_attempts.get(user.username, [])
        ),
        account_locked=len(auth_service._failed_login_attempts.get(user.username, []))
        >= 5,
    )


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: CreateUserRequest,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
) -> UserResponse:
    """Create a new user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    try:
        # Only superusers can create other superusers
        if user_data.is_superuser and not current_user.is_superuser:
            raise HTTPException(
                status_code=403, detail="Only superusers can create other superusers"
            )

        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            roles=user_data.roles,
        )

        # Update additional fields
        user.is_active = user_data.is_active
        user.is_superuser = user_data.is_superuser

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
            api_key_count=len(user.api_keys),
            failed_login_attempts=0,
            account_locked=False,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UpdateUserRequest,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
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

    # Only superusers can modify other superusers
    if user.is_superuser and not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="Only superusers can modify other superusers"
        )

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
        # Only superusers can change superuser status
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=403, detail="Only superusers can change superuser status"
            )
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
        api_key_count=len(user.api_keys),
        failed_login_attempts=len(
            auth_service._failed_login_attempts.get(user.username, [])
        ),
        account_locked=len(auth_service._failed_login_attempts.get(user.username, []))
        >= 5,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_super_admin),
) -> dict:
    """Delete a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-deletion
    current_user_obj = None
    if current_user:
        for u in auth_service._users.values():
            if u.username == current_user:
                current_user_obj = u
                break

    if current_user_obj and current_user_obj.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    # Only superusers can delete other superusers
    if user.is_superuser and not current_user_obj.is_superuser:
        raise HTTPException(
            status_code=403, detail="Only superusers can delete other superusers"
        )

    # Remove user and their API keys
    user = auth_service._users.pop(user_id)
    for api_key in user.api_keys:
        auth_service._api_keys.pop(api_key, None)

    return {"success": True, "message": f"User {user.username} deleted"}


@router.post(
    "/users/{user_id}/api-keys",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    user_id: str,
    api_key_data: CreateApiKeyRequest,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
) -> ApiKeyResponse:
    """Create an API key for a user. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    try:
        api_key = auth_service.create_api_key(user_id, api_key_data.key_name)

        # Get user for created_at timestamp
        user = auth_service._users.get(user_id)
        created_at = user.created_at.isoformat() if user else datetime.now().isoformat()

        return ApiKeyResponse(
            api_key=api_key,
            user_id=user_id,
            key_name=api_key_data.key_name,
            created_at=created_at,
            expires_at=None,  # API keys don't expire by default
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
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
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
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

    # Count users per role
    all_users = list(auth_service._users.values())
    role_counts = {}
    for role in ["admin", "user", "viewer"]:
        role_counts[role] = sum(1 for u in all_users if role in u.roles)

    roles = []
    for role in ["admin", "user", "viewer"]:
        permissions = auth_service._get_permissions_for_roles([role])
        roles.append(
            RoleInfo(
                role=role,
                permissions=permissions,
                description=role_descriptions.get(role, ""),
                user_count=role_counts.get(role, 0),
            )
        )

    return roles


@router.get("/permissions")
async def list_permissions(
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
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
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
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


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    request: PasswordResetRequest,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_super_admin),
) -> dict:
    """Reset user password (admin only). Requires super admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Reset password
    user.hashed_password = auth_service.hash_password(request.new_password)

    # Force password change on next login if requested
    if request.force_change_on_login:
        auth_service.force_password_reset(user_id)

    return {
        "success": True,
        "message": f"Password reset for user {user.username}",
        "force_change_on_login": request.force_change_on_login,
    }


@router.post("/users/{user_id}/unlock")
async def unlock_user_account(
    user_id: str,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_super_admin),
) -> dict:
    """Unlock user account after failed login attempts. Requires super admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Clear failed login attempts
    if user.username in auth_service._failed_login_attempts:
        del auth_service._failed_login_attempts[user.username]

    return {
        "success": True,
        "message": f"Account unlocked for user {user.username}",
    }


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
) -> SystemStatsResponse:
    """Get system statistics. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    users = list(auth_service._users.values())

    # Calculate statistics
    total_users = len(users)
    active_users = sum(1 for user in users if user.is_active)
    inactive_users = total_users - active_users
    admin_users = sum(1 for user in users if user.is_superuser or "admin" in user.roles)
    total_api_keys = sum(len(user.api_keys) for user in users)

    # Mock recent activity (in production, this would come from audit logs)
    recent_logins = sum(1 for user in users if user.last_login)
    failed_login_attempts = sum(
        len(attempts) for attempts in auth_service._failed_login_attempts.values()
    )

    return SystemStatsResponse(
        total_users=total_users,
        active_users=active_users,
        inactive_users=inactive_users,
        admin_users=admin_users,
        total_api_keys=total_api_keys,
        recent_logins=recent_logins,
        failed_login_attempts=failed_login_attempts,
        system_uptime="24h 30m",  # Mock uptime
    )


@router.get("/audit-log")
async def get_audit_log(
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_super_admin),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    event_type: str | None = Query(None, description="Filter by event type"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    start_date: str | None = Query(
        None, description="Filter by start date (ISO format)"
    ),
    end_date: str | None = Query(None, description="Filter by end date (ISO format)"),
) -> dict:
    """Get audit log entries. Requires super admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Mock audit log data (in production, this would come from a database)
    all_users = list(auth_service._users.values())
    mock_events = []

    # Generate mock audit events
    event_types = [
        "login",
        "logout",
        "password_reset",
        "role_change",
        "api_key_created",
        "api_key_revoked",
    ]

    for i in range(200):  # Generate 200 mock events
        user = all_users[i % len(all_users)] if all_users else None
        if user:
            event = {
                "event_id": f"audit_{i:06d}",
                "user_id": user.id,
                "username": user.username,
                "event_type": event_types[i % len(event_types)],
                "event_details": {
                    "action": "user_action",
                    "resource": "system",
                    "success": True,
                },
                "timestamp": datetime.now().isoformat(),
                "ip_address": f"192.168.1.{(i % 255) + 1}",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "admin_user_id": current_user.id
                if hasattr(current_user, "id")
                else None,
            }
            mock_events.append(event)

    # Apply filters
    filtered_events = mock_events
    if event_type:
        filtered_events = [e for e in filtered_events if e["event_type"] == event_type]
    if user_id:
        filtered_events = [e for e in filtered_events if e["user_id"] == user_id]

    # Pagination
    total = len(filtered_events)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_events = filtered_events[start_idx:end_idx]

    return {
        "events": paginated_events,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": end_idx < total,
        "has_previous": page > 1,
        "filters": {
            "event_type": event_type,
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date,
        },
    }


@router.get("/users/{user_id}/activity")
async def get_user_activity(
    user_id: str,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
) -> dict:
    """Get user activity summary. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Mock activity data (in production, this would come from audit logs)
    activity_summary = {
        "user_id": user_id,
        "username": user.username,
        "period_days": days,
        "login_count": 12,
        "api_requests": 145,
        "failed_login_attempts": len(
            auth_service._failed_login_attempts.get(user.username, [])
        ),
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "api_keys_used": len(user.api_keys),
        "resources_accessed": {
            "detectors": 25,
            "datasets": 15,
            "experiments": 8,
        },
        "activity_by_day": [
            {"date": "2024-01-01", "logins": 2, "api_requests": 15},
            {"date": "2024-01-02", "logins": 1, "api_requests": 22},
            {"date": "2024-01-03", "logins": 3, "api_requests": 18},
            {"date": "2024-01-04", "logins": 1, "api_requests": 31},
            {"date": "2024-01-05", "logins": 2, "api_requests": 24},
            {"date": "2024-01-06", "logins": 1, "api_requests": 19},
            {"date": "2024-01-07", "logins": 2, "api_requests": 16},
        ],
    }

    return activity_summary


@router.post("/users/{user_id}/sessions/invalidate")
async def invalidate_user_sessions(
    user_id: str,
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_super_admin),
) -> dict:
    """Invalidate all sessions for a user. Requires super admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    user = auth_service._users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Invalidate all sessions for the user
    count = auth_service.invalidate_user_sessions(user_id)

    return {
        "success": True,
        "message": f"Invalidated {count} sessions for user {user.username}",
        "sessions_invalidated": count,
    }


@router.get("/health")
async def admin_health_check(
    container: Container = Depends(get_container_simple),
    current_user: UserModel = Depends(require_tenant_admin),
) -> dict:
    """Admin health check endpoint. Requires admin permissions."""
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Check various system components
    system_health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "auth_service": "healthy" if auth_service else "unhealthy",
            "database": "healthy",  # Mock status
            "cache": "healthy",  # Mock status
            "api": "healthy",  # Mock status
        },
        "metrics": {
            "total_users": len(auth_service._users),
            "active_sessions": 25,  # Mock number
            "failed_login_attempts": sum(
                len(attempts)
                for attempts in auth_service._failed_login_attempts.values()
            ),
        },
    }

    # Determine overall status
    component_statuses = list(system_health["components"].values())
    if all(status == "healthy" for status in component_statuses):
        system_health["status"] = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        system_health["status"] = "unhealthy"
    else:
        system_health["status"] = "degraded"

    return system_health
