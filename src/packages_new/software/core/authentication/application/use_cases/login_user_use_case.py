"""
Login user use case
"""
from typing import Optional, Tuple
from uuid import UUID
from ..dto.login_request_dto import LoginRequestDto
from ..dto.login_response_dto import LoginResponseDto
from ..services.token_service import TokenService
from ...domain.entities.user import User
from ...domain.repositories.user_repository import UserRepository
from ...domain.services.authentication_service import AuthenticationService
from ...domain.value_objects.email import Email

class LoginUserUseCase:
    """Use case for user login"""
    
    def __init__(self, 
                 user_repository: UserRepository,
                 authentication_service: AuthenticationService,
                 token_service: TokenService):
        self.user_repository = user_repository
        self.authentication_service = authentication_service
        self.token_service = token_service
    
    def execute(self, request: LoginRequestDto) -> LoginResponseDto:
        """Execute login use case"""
        # Find user by email or username
        user = self._find_user(request.identifier)
        
        if not user:
            return LoginResponseDto(
                success=False,
                error="Invalid credentials",
                access_token=None,
                refresh_token=None,
                user_id=None
            )
        
        # Authenticate user
        if not self.authentication_service.authenticate_user(user, request.password):
            # Save updated user (with incremented failed attempts)
            self.user_repository.save(user)
            
            return LoginResponseDto(
                success=False,
                error="Invalid credentials",
                access_token=None,
                refresh_token=None,
                user_id=None
            )
        
        # Generate tokens
        access_token = self.token_service.generate_access_token(user)
        refresh_token = self.token_service.generate_refresh_token(user)
        
        # Save updated user (with reset failed attempts)
        self.user_repository.save(user)
        
        return LoginResponseDto(
            success=True,
            error=None,
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.id
        )
    
    def _find_user(self, identifier: str) -> Optional[User]:
        """Find user by email or username"""
        # Try to find by email first
        try:
            email = Email(identifier)
            return self.user_repository.find_by_email(email)
        except ValueError:
            # If not a valid email, try username
            return self.user_repository.find_by_username(identifier)