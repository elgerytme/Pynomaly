"""
Authentication API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from ...application.use_cases.login_user_use_case import LoginUserUseCase
from ...application.dto.login_request_dto import LoginRequestDto
from ...application.dto.login_response_dto import LoginResponseDto
from ...application.services.token_service import TokenService
from ...domain.repositories.user_repository import UserRepository
from ...domain.services.authentication_service import AuthenticationService

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Dependency injection (would be configured in main app)
def get_user_repository() -> UserRepository:
    # Implementation depends on infrastructure setup
    pass

def get_authentication_service() -> AuthenticationService:
    return AuthenticationService()

def get_token_service() -> TokenService:
    # Configuration would come from environment
    return TokenService(secret_key="your-secret-key")

def get_login_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    auth_service: AuthenticationService = Depends(get_authentication_service),
    token_service: TokenService = Depends(get_token_service)
) -> LoginUserUseCase:
    return LoginUserUseCase(user_repo, auth_service, token_service)

@router.post("/login", response_model=LoginResponseDto)
async def login(
    request: LoginRequestDto,
    login_use_case: LoginUserUseCase = Depends(get_login_use_case)
) -> LoginResponseDto:
    """
    User login endpoint
    
    Authenticates user credentials and returns JWT tokens
    """
    try:
        response = login_use_case.execute(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=response.error
            )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    token_service: TokenService = Depends(get_token_service)
):
    """
    User logout endpoint
    
    Invalidates the current access token
    """
    token = credentials.credentials
    
    if not token_service.is_token_valid(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    # In a real implementation, you would blacklist the token
    # For now, we just return success
    return {"message": "Logged out successfully"}

@router.post("/refresh")
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    token_service: TokenService = Depends(get_token_service),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Refresh access token endpoint
    
    Generates new access token using refresh token
    """
    refresh_token = credentials.credentials
    
    if not token_service.is_refresh_token(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = token_service.get_user_id_from_token(refresh_token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = user_repo.find_by_id(user_id)
    if not user or not user.can_login():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is not accessible"
        )
    
    # Generate new access token
    access_token = token_service.generate_access_token(user)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 3600
    }

@router.get("/me")
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    token_service: TokenService = Depends(get_token_service),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Get current user information
    
    Returns user information from valid access token
    """
    token = credentials.credentials
    
    if not token_service.is_access_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token"
        )
    
    user_id = token_service.get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token"
        )
    
    user = user_repo.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }