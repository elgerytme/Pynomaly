"""Authentication endpoints for API."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from pynomaly.infrastructure.security.rbac_middleware import require_auth

# from pydantic import EmailStr  # Temporarily disabled due to missing email-validator
from pynomaly.domain.exceptions import AuthenticationError
from pynomaly.infrastructure.auth import (
    JWTAuthService,
    TokenResponse,
    UserModel,
    get_auth,
)
from pynomaly.infrastructure.auth.middleware import get_current_user

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""

    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request."""

    username: str
    email: str  # EmailStr temporarily disabled
    password: str
    full_name: str | None = None


class UserResponse(BaseModel):
    """User response model."""

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


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
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

    try:
        user = auth_service.authenticate_user(form_data.username, form_data.password)
        return auth_service.create_access_token(user)

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_model=TokenResponse)
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
        return auth_service.refresh_access_token(refresh_token)

    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me", response_model=UserResponse)
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
    "/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED
)
async def create_api_key(
    request: APIKeyRequest,
    current_user: Annotated[UserModel | None, Depends(require_auth())],
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

        return APIKeyResponse(
            api_key=api_key,
            name=request.name,
            created_at=current_user.created_at.isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {e}",
        )


@router.delete("/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    return {"message": "API key revoked successfully"}


@router.post("/logout")
async def logout(
    current_user: Annotated[UserModel | None, Depends(get_current_user)],
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
    
    # In a production system, you might want to:
    # 1. Add the token to a blacklist
    # 2. Clear any server-side sessions
    # 3. Log the logout event

    return {"message": "Logged out successfully"}
