"""
FastAPI router for user management and multi-tenancy.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
import secrets
import hashlib

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from pynomaly.application.services.user_management_service import UserManagementService
from pynomaly.domain.entities.user import UserRole, TenantPlan, UserStatus, TenantStatus
from pynomaly.infrastructure.security.audit_logging import (
    get_audit_logger, AuditEventType, AuditSeverity
)
from pynomaly.shared.exceptions import (
    UserNotFoundError, TenantNotFoundError, ValidationError,
    AuthenticationError, AuthorizationError, ResourceLimitError
)
from pynomaly.shared.types import UserId, TenantId

# Router setup
router = APIRouter(prefix="/api/users", tags=["User Management"])
security = HTTPBearer()

# Request/Response Models
class CreateUserRequest(BaseModel):
    email: EmailStr
    username: str
    first_name: str
    last_name: str
    password: str
    tenant_id: Optional[UUID] = None
    role: UserRole = UserRole.VIEWER


class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    first_name: str
    last_name: str
    full_name: str
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    email_verified_at: Optional[datetime]
    tenant_roles: List[dict]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    user: UserResponse
    token: str
    expires_at: datetime


class CreateTenantRequest(BaseModel):
    name: str
    domain: str
    plan: TenantPlan = TenantPlan.FREE
    admin_email: Optional[EmailStr] = None
    admin_password: Optional[str] = None


class TenantResponse(BaseModel):
    id: UUID
    name: str
    domain: str
    plan: TenantPlan
    status: TenantStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    contact_email: str
    billing_email: str


class TenantUsageResponse(BaseModel):
    tenant_id: UUID
    tenant_name: str
    plan: str
    status: str
    usage: dict
    last_updated: str


class InviteUserRequest(BaseModel):
    email: EmailStr
    role: UserRole = UserRole.VIEWER


class UpdateUserRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None


class PasswordResetRequest(BaseModel):
    new_password: str = Field(..., min_length=8)


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


# Dependency injection
async def get_user_management_service() -> UserManagementService:
    """Get user management service instance."""
    # TODO: Implement proper dependency injection
    # For now, this is a placeholder
    from pynomaly.infrastructure.repositories.sqlalchemy_user_repository import (
        SQLAlchemyUserRepository, SQLAlchemyTenantRepository, SQLAlchemySessionRepository
    )
    from sqlalchemy.orm import sessionmaker
    
    # This should be injected via container
    session_factory = sessionmaker()  # Configure with actual database
    
    user_repo = SQLAlchemyUserRepository(session_factory)
    tenant_repo = SQLAlchemyTenantRepository(session_factory)
    session_repo = SQLAlchemySessionRepository(session_factory)
    
    return UserManagementService(user_repo, tenant_repo, session_repo)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Get current authenticated user."""
    try:
        user = await user_service.get_user_by_session(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def require_tenant_admin(
    tenant_id: UUID,
    current_user = Depends(get_current_user)
):
    """Require tenant admin permissions."""
    if not (current_user.is_super_admin() or 
            current_user.has_role_in_tenant(TenantId(str(tenant_id)), UserRole.TENANT_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


# Authentication Endpoints
@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Authenticate user and return access token."""
    try:
        user, token = await user_service.authenticate_user(request.email, request.password)
        
        return LoginResponse(
            user=UserResponse(
                id=UUID(user.id),
                email=user.email,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                full_name=user.full_name,
                status=user.status,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login_at=user.last_login_at,
                email_verified_at=user.email_verified_at,
                tenant_roles=[
                    {
                        "tenant_id": str(tr.tenant_id),
                        "role": tr.role.value,
                        "granted_at": tr.granted_at.isoformat(),
                        "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                    }
                    for tr in user.tenant_roles
                ]
            ),
            token=token,
            expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59)
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/auth/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Logout user and invalidate session."""
    try:
        await user_service.logout_user(credentials.credentials)
        return {"message": "Successfully logged out"}
    except Exception:
        # Always return success for logout
        return {"message": "Successfully logged out"}


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information."""
    audit_logger = get_audit_logger()
    audit_logger.log_event(
        AuditEventType.DATA_ACCESSED,
        user_id=str(current_user.id),
        outcome="success",
        severity=AuditSeverity.LOW,
        details={"action": "view_own_profile"}
    )
    
    return UserResponse(
        id=UUID(current_user.id),
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        full_name=current_user.full_name,
        status=current_user.status,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login_at=current_user.last_login_at,
        email_verified_at=current_user.email_verified_at,
        tenant_roles=[
            {
                "tenant_id": str(tr.tenant_id),
                "role": tr.role.value,
                "granted_at": tr.granted_at.isoformat(),
                "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
            }
            for tr in current_user.tenant_roles
        ]
    )


# User Management Endpoints
@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Create a new user."""
    audit_logger = get_audit_logger()
    try:
        user = await user_service.create_user(
            email=request.email,
            username=request.username,
            first_name=request.first_name,
            last_name=request.last_name,
            password=request.password,
            tenant_id=TenantId(str(request.tenant_id)) if request.tenant_id else None,
            role=request.role
        )

        # Log user creation
        audit_logger.log_event(
            AuditEventType.USER_CREATED,
            user_id=str(user.id),
            outcome="success",
            severity=AuditSeverity.LOW,
            details={"email": user.email}
        )
        
        return UserResponse(
            id=UUID(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            tenant_roles=[
                {
                    "tenant_id": str(tr.tenant_id),
                    "role": tr.role.value,
                    "granted_at": tr.granted_at.isoformat(),
                    "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                }
                for tr in user.tenant_roles
            ]
        )
    except (ValidationError, ResourceLimitError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[UserResponse])
async def list_users(
    tenant_id: Optional[UUID] = None,
    status: Optional[UserStatus] = None,
    role: Optional[UserRole] = None,
    limit: int = 100,
    offset: int = 0,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """List users with optional filters."""
    audit_logger = get_audit_logger()
    try:
        users = await user_service.list_users(
            tenant_id=TenantId(str(tenant_id)) if tenant_id else None,
            status=status,
            role=role,
            limit=limit,
            offset=offset
        )

        audit_logger.log_event(
            AuditEventType.DATA_ACCESSED,
            user_id=str(current_user.id),
            outcome="success",
            severity=AuditSeverity.LOW,
            details={
                "action": "list_users",
                "filters": {
                    "tenant_id": str(tenant_id) if tenant_id else None,
                    "status": status,
                    "role": role,
                    "limit": limit,
                    "offset": offset
                },
                "result_count": len(users)
            }
        )
        
        return [
            UserResponse(
                id=UUID(user.id),
                email=user.email,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                full_name=user.full_name,
                status=user.status,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login_at=user.last_login_at,
                email_verified_at=user.email_verified_at,
                tenant_roles=[
                    {
                        "tenant_id": str(tr.tenant_id),
                        "role": tr.role.value,
                        "granted_at": tr.granted_at.isoformat(),
                        "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                    }
                    for tr in user.tenant_roles
                ]
            )
            for user in users
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Get user by ID."""
    audit_logger = get_audit_logger()
    try:
        user = await user_service._user_repo.get_user_by_id(UserId(str(user_id)))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if current user can view this user
        # (same tenant or super admin)
        if not current_user.is_super_admin():
            common_tenants = set(current_user.get_tenant_ids()) & set(user.get_tenant_ids())
            if not common_tenants:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )

        audit_logger.log_event(
            AuditEventType.DATA_ACCESSED,
            user_id=str(current_user.id),
            outcome="success",
            severity=AuditSeverity.LOW,
            details={
                "action": "view_user",
                "target_user_id": str(user_id)
            }
        )
        
        return UserResponse(
            id=UUID(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            tenant_roles=[
                {
                    "tenant_id": str(tr.tenant_id),
                    "role": tr.role.value,
                    "granted_at": tr.granted_at.isoformat(),
                    "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                }
                for tr in user.tenant_roles
            ]
        )
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.put("/{user_id}/status", response_model=UserResponse)
async def toggle_user_status(
    user_id: UUID,
    new_status: UserStatus,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Toggle user status (activate/deactivate)."""
    audit_logger = get_audit_logger()
    try:
        user = await user_service.toggle_user_status(UserId(str(user_id)), new_status)

        audit_logger.log_event(
            AuditEventType.USER_ACTIVATED if new_status == UserStatus.ACTIVE else AuditEventType.USER_DEACTIVATED,
            user_id=str(current_user.id),
            outcome="success",
            severity=AuditSeverity.MEDIUM,
            details={
                "target_user_id": str(user_id),
                "new_status": new_status.value,
                "previous_status": "unknown"  # Could be enhanced to track previous status
            }
        )

        return UserResponse(
            id=UUID(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            tenant_roles=[
                {
                    "tenant_id": str(tr.tenant_id),
                    "role": tr.role.value,
                    "granted_at": tr.granted_at.isoformat(),
                    "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                }
                for tr in user.tenant_roles
            ]
        )
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Update user details."""
    audit_logger = get_audit_logger()
    try:
        user = await user_service.update_user(user_id=UserId(str(user_id)), update_data=request.dict(exclude_unset=True))

        audit_logger.log_event(
            AuditEventType.USER_UPDATED,
            user_id=str(user.id),
            outcome="success",
            severity=AuditSeverity.LOW,
            details=request.dict(exclude_unset=True)
        )

        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            tenant_roles=[
                {
                    "tenant_id": str(tr.tenant_id),
                    "role": tr.role.value,
                    "granted_at": tr.granted_at.isoformat(),
                    "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                }
                for tr in user.tenant_roles
            ]
        )
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.put("/{user_id}/reset-password", response_model=dict)
async def reset_password(
    user_id: UUID,
    request: PasswordResetRequest,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Reset user password."""
    audit_logger = get_audit_logger()
    try:
        await user_service.reset_password(UserId(str(user_id)), request.new_password)

        audit_logger.log_event(
            AuditEventType.PASSWORD_CHANGED,
            user_id=str(user_id),
            outcome="success",
            severity=AuditSeverity.MEDIUM,
        )

        return {"message": "Password reset successfully."}
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Tenant Management Endpoints
@router.post("/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: CreateTenantRequest,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Create a new tenant (super admin only)."""
    if not current_user.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can create tenants"
        )
    
    try:
        tenant, admin_user = await user_service.create_tenant(
            name=request.name,
            domain=request.domain,
            plan=request.plan,
            admin_email=request.admin_email or "",
            admin_password=request.admin_password or ""
        )
        
        return TenantResponse(
            id=UUID(tenant.id),
            name=tenant.name,
            domain=tenant.domain,
            plan=tenant.plan,
            status=tenant.status,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            expires_at=tenant.expires_at,
            contact_email=tenant.contact_email,
            billing_email=tenant.billing_email
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/tenants/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: UUID,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Get tenant information."""
    # Check if user has access to this tenant
    if not (current_user.is_super_admin() or 
            current_user.has_role_in_tenant(TenantId(str(tenant_id)), UserRole.TENANT_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    try:
        tenant = await user_service._tenant_repo.get_tenant_by_id(TenantId(str(tenant_id)))
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return TenantResponse(
            id=UUID(tenant.id),
            name=tenant.name,
            domain=tenant.domain,
            plan=tenant.plan,
            status=tenant.status,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            expires_at=tenant.expires_at,
            contact_email=tenant.contact_email,
            billing_email=tenant.billing_email
        )
    except TenantNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )


@router.get("/tenants/{tenant_id}/usage", response_model=TenantUsageResponse)
async def get_tenant_usage(
    tenant_id: UUID,
    current_user = Depends(require_tenant_admin),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Get tenant usage statistics."""
    try:
        usage_report = await user_service.get_tenant_usage_report(TenantId(str(tenant_id)))
        return TenantUsageResponse(**usage_report)
    except TenantNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )


@router.get("/tenants/{tenant_id}/users", response_model=List[UserResponse])
async def get_tenant_users(
    tenant_id: UUID,
    current_user = Depends(require_tenant_admin),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Get all users in a tenant."""
    try:
        users = await user_service._user_repo.get_users_by_tenant(TenantId(str(tenant_id)))
        
        return [
            UserResponse(
                id=UUID(user.id),
                email=user.email,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                full_name=user.full_name,
                status=user.status,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login_at=user.last_login_at,
                email_verified_at=user.email_verified_at,
                tenant_roles=[
                    {
                        "tenant_id": str(tr.tenant_id),
                        "role": tr.role.value,
                        "granted_at": tr.granted_at.isoformat(),
                        "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                    }
                    for tr in user.tenant_roles
                ]
            )
            for user in users
        ]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant users"
        )


@router.post("/tenants/{tenant_id}/invite", response_model=UserResponse)
async def invite_user_to_tenant(
    tenant_id: UUID,
    request: InviteUserRequest,
    current_user = Depends(require_tenant_admin),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Invite a user to join a tenant."""
    try:
        user = await user_service.invite_user_to_tenant(
            inviter_id=UserId(current_user.id),
            tenant_id=TenantId(str(tenant_id)),
            email=request.email,
            role=request.role
        )
        
        return UserResponse(
            id=UUID(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            email_verified_at=user.email_verified_at,
            tenant_roles=[
                {
                    "tenant_id": str(tr.tenant_id),
                    "role": tr.role.value,
                    "granted_at": tr.granted_at.isoformat(),
                    "expires_at": tr.expires_at.isoformat() if tr.expires_at else None
                }
                for tr in user.tenant_roles
            ]
        )
    except (ValidationError, ResourceLimitError, AuthorizationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/tenants/{tenant_id}/users/{user_id}/role", response_model=dict)
async def update_user_role_in_tenant(
    tenant_id: UUID,
    user_id: UUID,
    request: UpdateUserRoleRequest,
    current_user = Depends(require_tenant_admin),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Update a user's role in a tenant."""
    try:
        tenant_role = await user_service._user_repo.update_user_role_in_tenant(
            user_id=UserId(str(user_id)),
            tenant_id=TenantId(str(tenant_id)),
            role=request.role
        )
        
        return {
            "message": "User role updated successfully",
            "user_id": str(user_id),
            "tenant_id": str(tenant_id),
            "new_role": request.role.value
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{user_id}", response_model=dict)
async def delete_user(
    user_id: UUID,
    current_user = Depends(get_current_user),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Delete user by ID."""
    audit_logger = get_audit_logger()
    try:
        await user_service.delete_user(UserId(str(user_id)))

        audit_logger.log_event(
            AuditEventType.USER_DELETED,
            user_id=str(user_id),
            outcome="success",
            severity=AuditSeverity.HIGH,
        )

        return {"message": "User deleted successfully."}
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.delete("/tenants/{tenant_id}/users/{user_id}")
async def remove_user_from_tenant(
    tenant_id: UUID,
    user_id: UUID,
    current_user = Depends(require_tenant_admin),
    user_service: UserManagementService = Depends(get_user_management_service)
):
    """Remove a user from a tenant."""
    audit_logger = get_audit_logger()
    try:
        success = await user_service._user_repo.remove_user_from_tenant(
            user_id=UserId(str(user_id)),
            tenant_id=TenantId(str(tenant_id))
        )
        
        if success:
            audit_logger.log_event(
                AuditEventType.USER_DELETED,
                user_id=str(user_id),
                outcome="success",
                severity=AuditSeverity.MEDIUM,
                details={"tenant_id": str(tenant_id)}
            )

            # Update tenant usage
            await user_service._tenant_repo.update_tenant_usage(
                TenantId(str(tenant_id)), {"users_count": "-1"}
            )
            
            return {"message": "User removed from tenant successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in tenant"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
