"""Authentication endpoints for API with comprehensive audit logging."""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

# from pydantic import EmailStr  # Temporarily disabled due to missing email-validator
from interfaces.domain.exceptions import AuthenticationError
from interfaces.domain.services.mfa_service import MFAService
from interfaces.infrastructure.auth import (
    JWTAuthService,
    TokenResponse,
    UserModel,
    get_auth,
)
from interfaces.infrastructure.auth.middleware import get_current_user
from interfaces.infrastructure.cache import get_cache
from interfaces.infrastructure.security.audit_logger import (
    AuditLevel,
    SecurityEventType,
    audit_context,
    get_audit_logger,
)
from interfaces.infrastructure.security.rbac_middleware import require_auth
from monorepo.infrastructure.services.email_service import get_email_service

router = APIRouter()


def get_mfa_service() -> MFAService:
    """Get MFA service instance."""
    cache = get_cache()
    return MFAService(redis_client=cache)


class LoginRequest(BaseModel):
    """Login request processor."""

    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request."""

    username: str
    email: str  # EmailStr temporarily disabled
    password: str
    full_name: str | None = None


class UserResponse(BaseModel):
    """User response processor."""

    id: str
    username: str
    email: str
    full_name: str | None
    is_active: bool
    roles: list[str]
    created_at: str


class APIKeyRequest(BaseModel):
    """API key creation request."""

    name: str
    description: str | None = None


class APIKeyResponse(BaseModel):
    """API key response."""

    api_key: str
    name: str
    created_at: str


class PasswordResetRequest(BaseModel):
    """Password reset request processor."""

    email: str  # Email address to send reset link


class PasswordResetResponse(BaseModel):
    """Password reset response processor."""

    message: str
    email: str


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request processor."""

    token: str
    new_password: str


class PasswordResetConfirmResponse(BaseModel):
    """Password reset confirmation response processor."""

    message: str


class MFAChallengeResponse(BaseModel):
    """MFA challenge response processor."""

    mfa_required: bool
    challenge_id: str
    available_methods: list[str]
    message: str


class MFAVerificationRequest(BaseModel):
    """MFA verification request processor."""

    challenge_id: str
    method_type: str
    verification_code: str
    remember_device: bool = False


@router.post("/login", response_processor=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> TokenResponse:
    """Login with username and password.

    Args:
        form_data: OAuth2 form data
        auth_service: Auth service

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If authentication fails
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    # Extract client information
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    try:
        user = await auth_service.authenticate_user(
            form_data.username, form_data.password
        )

        # Log successful login
        audit_logger = get_audit_logger()
        with audit_context(
            correlation_id=f"login_{user.id}_{int(time.time())}",
            user_id=user.id,
            ip_address=client_ip,
            user_agent=user_agent,
        ):
            audit_logger.log_security_event(
                SecurityEventType.AUTH_LOGIN_SUCCESS,
                f"User {user.username} logged in successfully",
                level=AuditLevel.INFO,
                details={
                    "user_id": user.id,
                    "username": user.username,
                    "ip_address": client_ip,
                    "user_agent": user_agent,
                },
            )

        return auth_service.create_access_token(user, client_ip, user_agent)

    except AuthenticationError as e:
        # Log failed login
        audit_logger = get_audit_logger()
        with audit_context(
            correlation_id=f"login_fail_{form_data.username}_{int(time.time())}",
            ip_address=client_ip,
            user_agent=user_agent,
        ):
            audit_logger.log_security_event(
                SecurityEventType.AUTH_LOGIN_FAILURE,
                f"Login failed for user {form_data.username}: {str(e)}",
                level=AuditLevel.WARNING,
                details={
                    "username": form_data.username,
                    "ip_address": client_ip,
                    "user_agent": user_agent,
                    "failure_reason": str(e),
                },
                risk_score=60,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_processor=TokenResponse)
async def refresh_token(
    refresh_token: str,
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> TokenResponse:
    """Refresh access token using refresh token.

    Args:
        refresh_token: Refresh token
        auth_service: Auth service

    Returns:
        New access token

    Raises:
        HTTPException: If refresh fails
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        result = auth_service.refresh_access_token(refresh_token)

        # Log token refresh
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_TOKEN_REFRESH,
            "Access token refreshed successfully",
            level=AuditLevel.INFO,
            details={"refresh_token_used": True},
        )

        return result

    except AuthenticationError as e:
        # Log failed token refresh
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_TOKEN_REFRESH,
            f"Token refresh failed: {str(e)}",
            level=AuditLevel.WARNING,
            details={"failure_reason": str(e)},
            risk_score=40,
        )

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post(
    "/register", response_processor=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    request_data: RegisterRequest,
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> UserResponse:
    """Register a new user.

    Args:
        request_data: Registration data
        auth_service: Auth service

    Returns:
        Created user

    Raises:
        HTTPException: If registration fails
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        user = auth_service.create_user(
            username=request_data.username,
            email=request_data.email,
            password=request_data.password,
            full_name=request_data.full_name,
        )

        # Log user registration
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_LOGIN_SUCCESS,  # Registration is similar to login
            f"New user registered: {user.username}",
            level=AuditLevel.INFO,
            details={
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
            },
        )

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            roles=user.roles,
            created_at=user.created_at.isoformat(),
        )

    except ValueError as e:
        # Log failed registration
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,  # Registration failure
            f"User registration failed for {request_data.username}: {str(e)}",
            level=AuditLevel.WARNING,
            details={
                "username": request_data.username,
                "email": request_data.email,
                "failure_reason": str(e),
            },
            risk_score=30,
        )

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me", response_processor=UserResponse)
async def get_current_user_profile(
    current_user: Annotated[UserModel | None, Depends(require_auth())],
) -> UserResponse:
    """Get current user profile.

    Args:
        current_user: Current authenticated user

    Returns:
        User profile

    Raises:
        HTTPException: If no user is authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        roles=current_user.roles,
        created_at=current_user.created_at.isoformat(),
    )


@router.post(
    "/api-keys", response_processor=APIKeyResponse, status_code=status.HTTP_201_CREATED
)
async def create_api_key(
    request: APIKeyRequest,
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> APIKeyResponse:
    """Create a new API key for the current user.

    Args:
        request: API key details
        current_user: Current user
        auth_service: Auth service

    Returns:
        Created API key

    Raises:
        HTTPException: If creation fails
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        api_key = auth_service.create_api_key(
            user_id=current_user.id, key_name=request.name
        )

        # Log API key creation
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.API_KEY_CREATED,
            f"API key created for user {current_user.username}",
            level=AuditLevel.INFO,
            details={
                "user_id": current_user.id,
                "username": current_user.username,
                "api_key_name": request.name,
            },
        )

        return APIKeyResponse(
            api_key=api_key,
            name=request.name,
            created_at=current_user.created_at.isoformat(),
        )

    except Exception as e:
        # Log API key creation failure
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.API_KEY_CREATED,
            f"API key creation failed for user {current_user.username}: {str(e)}",
            level=AuditLevel.ERROR,
            details={
                "user_id": current_user.id,
                "username": current_user.username,
                "api_key_name": request.name,
                "failure_reason": str(e),
            },
            risk_score=40,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {e}",
        )


@router.delete("/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    current_user: Annotated[Any | None, Depends(get_current_user)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> dict:
    """Revoke an API key.

    Args:
        api_key: API key to revoke
        current_user: Current user
        auth_service: Auth service

    Returns:
        Success message

    Raises:
        HTTPException: If revocation fails
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    # Check if API key belongs to current user
    if api_key not in current_user.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key not found or does not belong to current user",
        )

    success = auth_service.revoke_api_key(api_key)

    if not success:
        # Log API key revocation failure
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.API_KEY_REVOKED,
            f"API key revocation failed for user {current_user.username}: key not found",
            level=AuditLevel.WARNING,
            details={
                "user_id": current_user.id,
                "username": current_user.username,
                "api_key_prefix": api_key[:8] + "...",
                "failure_reason": "API key not found",
            },
            risk_score=30,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Log successful API key revocation
    audit_logger = get_audit_logger()
    audit_logger.log_security_event(
        SecurityEventType.API_KEY_REVOKED,
        f"API key revoked for user {current_user.username}",
        level=AuditLevel.INFO,
        details={
            "user_id": current_user.id,
            "username": current_user.username,
            "api_key_prefix": api_key[:8] + "...",
        },
    )

    return {"message": "API key revoked successfully"}


@router.get("/api-keys")
async def list_api_keys(
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
) -> dict:
    """List API keys for current user.

    Args:
        current_user: Current user

    Returns:
        List of API keys (without the actual key values)

    Raises:
        HTTPException: If user not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Return API keys without the actual key values for security
    api_keys = [
        {
            "key_id": f"key_{i}",
            "name": f"API Key {i}",
            "created_at": current_user.created_at.isoformat(),
            "last_used": None,
            "active": True,
        }
        for i, key in enumerate(current_user.api_keys, 1)
    ]

    return {"api_keys": api_keys, "count": len(api_keys)}


@router.post("/logout")
async def logout(
    current_user: Annotated[Any | None, Depends(get_current_user)],
) -> dict:
    """Logout current user.

    Note: With JWT, logout is typically handled client-side by removing the token.
    This endpoint can be used for server-side token blacklisting if implemented.

    Args:
        current_user: Current user

    Returns:
        Success message
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Log logout event
    audit_logger = get_audit_logger()
    audit_logger.log_security_event(
        SecurityEventType.AUTH_LOGOUT,
        f"User {current_user.username} logged out",
        level=AuditLevel.INFO,
        details={
            "user_id": current_user.id,
            "username": current_user.username,
        },
    )

    # In a production system, you might want to:
    # 1. Add the token to a blacklist
    # 2. Clear any server-side sessions
    # 3. Log the logout event (done above)

    return {"message": "Logged out successfully"}


@router.post("/password-reset", response_processor=PasswordResetResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> PasswordResetResponse:
    """Request password reset for user.

    Args:
        request: Password reset request
        auth_service: Auth service

    Returns:
        Password reset response

    Raises:
        HTTPException: If service is unavailable
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        # Generate password reset token
        reset_token = auth_service.create_password_reset_token(request.email)

        # Log password reset request
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_PASSWORD_CHANGE,
            f"Password reset requested for email {request.email}",
            level=AuditLevel.INFO,
            details={
                "email": request.email,
                "reset_method": "email_token",
            },
        )

        # Send email with reset link
        email_service = get_email_service()
        if email_service:
            try:
                email_sent = await email_service.send_password_reset_email(
                    email=request.email,
                    reset_token=reset_token,
                    user_name="User",  # Could be improved to get actual user name
                )
                if not email_sent:
                    # Log warning but don't fail - fallback to success message
                    audit_logger.log_security_event(
                        SecurityEventType.AUTH_PASSWORD_CHANGE,
                        f"Failed to send password reset email to {request.email}",
                        level=AuditLevel.WARNING,
                        details={"email": request.email, "error": "email_send_failed"},
                    )
            except Exception as e:
                # Log error but don't fail - fallback to success message
                audit_logger.log_security_event(
                    SecurityEventType.AUTH_PASSWORD_CHANGE,
                    f"Error sending password reset email to {request.email}: {str(e)}",
                    level=AuditLevel.ERROR,
                    details={"email": request.email, "error": str(e)},
                )
        else:
            # Email service not configured - log warning
            audit_logger.log_security_event(
                SecurityEventType.AUTH_PASSWORD_CHANGE,
                f"Email service not configured - password reset email not sent to {request.email}",
                level=AuditLevel.WARNING,
                details={
                    "email": request.email,
                    "error": "email_service_not_configured",
                },
            )

        return PasswordResetResponse(
            message="Password reset email sent successfully",
            email=request.email,
        )

    except Exception:
        # Log password reset failure (but don't reveal if email exists)
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_PASSWORD_CHANGE,
            f"Password reset request failed for email {request.email}",
            level=AuditLevel.WARNING,
            details={
                "email": request.email,
                "failure_reason": "User not found or error occurred",
            },
            risk_score=20,
        )

        # For security, don't reveal whether the email exists or not
        return PasswordResetResponse(
            message="If the email address exists, a password reset link has been sent",
            email=request.email,
        )


@router.post("/password-reset/confirm", response_processor=PasswordResetConfirmResponse)
async def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> PasswordResetConfirmResponse:
    """Confirm password reset with token.

    Args:
        request: Password reset confirmation request
        auth_service: Auth service

    Returns:
        Password reset confirmation response

    Raises:
        HTTPException: If token is invalid or service unavailable
    """
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        # Verify token and reset password
        success = auth_service.reset_password_with_token(
            request.token, request.new_password
        )

        # Log successful password reset
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_PASSWORD_CHANGE,
            "Password reset completed successfully",
            level=AuditLevel.INFO,
            details={
                "reset_method": "token_confirmation",
                "token_used": True,
            },
        )

        return PasswordResetConfirmResponse(
            message="Password reset successfully",
        )

    except ValueError as e:
        # Log password reset failure
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_PASSWORD_CHANGE,
            f"Password reset failed: {str(e)}",
            level=AuditLevel.WARNING,
            details={
                "reset_method": "token_confirmation",
                "failure_reason": str(e),
            },
            risk_score=50,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Log password reset error
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            SecurityEventType.AUTH_PASSWORD_CHANGE,
            f"Password reset error: {str(e)}",
            level=AuditLevel.ERROR,
            details={
                "reset_method": "token_confirmation",
                "error": str(e),
            },
            risk_score=60,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        )


@router.get("/sessions")
async def get_user_sessions(
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> dict:
    """Get active sessions for current user.

    Args:
        current_user: Current authenticated user
        auth_service: Auth service

    Returns:
        List of active sessions

    Raises:
        HTTPException: If user not authenticated or service unavailable
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        sessions = auth_service.get_user_sessions(current_user.id)
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sessions: {e}",
        )


@router.delete("/sessions")
async def invalidate_all_sessions(
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> dict:
    """Invalidate all sessions for current user.

    Args:
        current_user: Current authenticated user
        auth_service: Auth service

    Returns:
        Number of sessions invalidated

    Raises:
        HTTPException: If user not authenticated or service unavailable
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        count = auth_service.invalidate_user_sessions(current_user.id)
        return {"message": f"Invalidated {count} sessions", "count": count}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate sessions: {e}",
        )


@router.delete("/sessions/{session_id}")
async def invalidate_session(
    session_id: str,
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
) -> dict:
    """Invalidate a specific session.

    Args:
        session_id: Session ID to invalidate
        current_user: Current authenticated user
        auth_service: Auth service

    Returns:
        Success message

    Raises:
        HTTPException: If user not authenticated or service unavailable
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )

    try:
        success = auth_service._invalidate_session(session_id)
        if success:
            return {"message": "Session invalidated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate session: {e}",
        )
