"""
Enterprise Authentication Service

This service handles all authentication-related operations including
SSO, SAML, OAuth2, and local authentication with multi-tenancy support.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
from structlog import get_logger

from ..dto.auth_dto import (
    LoginRequest, LoginResponse, RegisterRequest, RegisterResponse,
    PasswordResetRequest, MFASetupRequest, MFAVerifyRequest,
    SAMLAuthRequest, OAuth2AuthRequest, RefreshTokenRequest
)
from ...domain.entities.user import User, UserSession, UserStatus, AuthProvider
from ...domain.entities.tenant import Tenant, TenantStatus
from ...domain.repositories.user_repository import UserRepository
from ...domain.repositories.tenant_repository import TenantRepository
from ...domain.repositories.session_repository import SessionRepository
from ...domain.services.password_service import PasswordService
from ...domain.services.mfa_service import MFAService
from ...infrastructure.external.sso_providers import SAMLProvider, OAuth2Provider

logger = get_logger(__name__)


class AuthConfig(BaseModel):
    """Authentication service configuration."""
    
    jwt_secret_key: str = Field(..., description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=60, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=30, description="Refresh token expiry")
    
    # Password settings
    min_password_length: int = Field(default=8)
    require_password_uppercase: bool = Field(default=True)
    require_password_lowercase: bool = Field(default=True)
    require_password_numbers: bool = Field(default=True)
    require_password_special: bool = Field(default=True)
    
    # Account lockout settings
    max_failed_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=30)
    
    # Session settings
    max_concurrent_sessions: int = Field(default=5)
    session_timeout_minutes: int = Field(default=480)  # 8 hours
    
    # MFA settings
    mfa_issuer_name: str = Field(default="anomaly_detection Enterprise")
    mfa_token_lifetime_seconds: int = Field(default=300)  # 5 minutes


class AuthService:
    """
    Enterprise Authentication Service
    
    Provides comprehensive authentication capabilities including:
    - Local username/password authentication
    - SSO integration (SAML, OAuth2, OIDC)
    - Multi-factor authentication
    - Session management
    - Password policies
    - Account lockout protection
    """
    
    def __init__(
        self,
        user_repository: UserRepository,
        tenant_repository: TenantRepository,
        session_repository: SessionRepository,
        password_service: PasswordService,
        mfa_service: MFAService,
        config: AuthConfig,
        saml_provider: Optional[SAMLProvider] = None,
        oauth2_provider: Optional[OAuth2Provider] = None
    ):
        self.user_repo = user_repository
        self.tenant_repo = tenant_repository
        self.session_repo = session_repository
        self.password_service = password_service
        self.mfa_service = mfa_service
        self.config = config
        self.saml_provider = saml_provider
        self.oauth2_provider = oauth2_provider
        
        # Initialize password context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        logger.info("AuthService initialized", config=config.dict())
    
    async def authenticate_local(
        self, 
        request: LoginRequest, 
        ip_address: Optional[str] = None
    ) -> LoginResponse:
        """
        Authenticate user with local credentials.
        
        Args:
            request: Login request with email/password
            ip_address: Client IP address for audit logging
            
        Returns:
            LoginResponse with tokens and user info
            
        Raises:
            AuthenticationError: If authentication fails
        """
        logger.info("Attempting local authentication", email=request.email)
        
        try:
            # Find user by email
            user = await self.user_repo.get_by_email(request.email)
            if not user:
                logger.warning("User not found", email=request.email)
                raise AuthenticationError("Invalid credentials")
            
            # Check tenant status
            tenant = await self.tenant_repo.get_by_id(user.tenant_id)
            if not tenant or not tenant.is_active():
                logger.warning("Tenant inactive", tenant_id=user.tenant_id)
                raise AuthenticationError("Account suspended")
            
            # Check user status
            if not user.is_active():
                logger.warning("User account inactive", user_id=user.id, status=user.status)
                raise AuthenticationError("Account suspended")
            
            # Check account lockout
            if user.is_locked():
                logger.warning("User account locked", user_id=user.id)
                raise AuthenticationError("Account locked due to failed login attempts")
            
            # Verify password
            if not self.password_service.verify_password(request.password, user.password_hash):
                await self._handle_failed_login(user, ip_address)
                logger.warning("Invalid password", user_id=user.id)
                raise AuthenticationError("Invalid credentials")
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not request.mfa_code:
                    logger.info("MFA required", user_id=user.id)
                    return LoginResponse(
                        success=False,
                        requires_mfa=True,
                        message="MFA verification required"
                    )
                
                if not self.mfa_service.verify_code(user.mfa_secret, request.mfa_code):
                    logger.warning("Invalid MFA code", user_id=user.id)
                    raise AuthenticationError("Invalid MFA code")
            
            # Authentication successful
            await self._handle_successful_login(user, ip_address)
            
            # Create session and tokens
            tokens = await self._create_user_session(user, ip_address, request.device_info)
            
            logger.info("Local authentication successful", user_id=user.id)
            
            return LoginResponse(
                success=True,
                user_id=user.id,
                tenant_id=user.tenant_id,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                user_info={
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "roles": list(user.roles),
                    "permissions": list(user.permissions)
                }
            )
            
        except Exception as e:
            logger.error("Authentication failed", error=str(e), email=request.email)
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError("Authentication failed")
    
    async def authenticate_saml(
        self, 
        request: SAMLAuthRequest, 
        ip_address: Optional[str] = None
    ) -> LoginResponse:
        """
        Authenticate user with SAML SSO.
        
        Args:
            request: SAML authentication request
            ip_address: Client IP address
            
        Returns:
            LoginResponse with tokens and user info
        """
        logger.info("Attempting SAML authentication", tenant_id=request.tenant_id)
        
        if not self.saml_provider:
            raise AuthenticationError("SAML authentication not configured")
        
        try:
            # Validate SAML response
            saml_data = self.saml_provider.validate_response(
                request.saml_response,
                request.relay_state
            )
            
            # Extract user information from SAML
            email = saml_data.get("email")
            if not email:
                raise AuthenticationError("Email not provided in SAML response")
            
            # Find or create user
            user = await self._get_or_create_sso_user(
                email=email,
                first_name=saml_data.get("first_name", ""),
                last_name=saml_data.get("last_name", ""),
                tenant_id=request.tenant_id,
                auth_provider=AuthProvider.SAML,
                external_id=saml_data.get("user_id")
            )
            
            # Create session and tokens
            tokens = await self._create_user_session(user, ip_address)
            
            logger.info("SAML authentication successful", user_id=user.id)
            
            return LoginResponse(
                success=True,
                user_id=user.id,
                tenant_id=user.tenant_id,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                user_info={
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "roles": list(user.roles),
                    "permissions": list(user.permissions)
                }
            )
            
        except Exception as e:
            logger.error("SAML authentication failed", error=str(e))
            raise AuthenticationError("SAML authentication failed")
    
    async def authenticate_oauth2(
        self, 
        request: OAuth2AuthRequest, 
        ip_address: Optional[str] = None
    ) -> LoginResponse:
        """
        Authenticate user with OAuth2/OIDC.
        
        Args:
            request: OAuth2 authentication request
            ip_address: Client IP address
            
        Returns:
            LoginResponse with tokens and user info
        """
        logger.info("Attempting OAuth2 authentication", 
                   provider=request.provider, tenant_id=request.tenant_id)
        
        if not self.oauth2_provider:
            raise AuthenticationError("OAuth2 authentication not configured")
        
        try:
            # Exchange authorization code for tokens
            oauth_tokens = await self.oauth2_provider.exchange_code(
                request.code,
                request.redirect_uri,
                request.provider
            )
            
            # Get user info from OAuth2 provider
            user_info = await self.oauth2_provider.get_user_info(
                oauth_tokens["access_token"],
                request.provider
            )
            
            email = user_info.get("email")
            if not email:
                raise AuthenticationError("Email not provided by OAuth2 provider")
            
            # Find or create user
            user = await self._get_or_create_sso_user(
                email=email,
                first_name=user_info.get("given_name", ""),
                last_name=user_info.get("family_name", ""),
                tenant_id=request.tenant_id,
                auth_provider=AuthProvider.OAUTH2,
                external_id=user_info.get("sub")
            )
            
            # Create session and tokens
            tokens = await self._create_user_session(user, ip_address)
            
            logger.info("OAuth2 authentication successful", user_id=user.id)
            
            return LoginResponse(
                success=True,
                user_id=user.id,
                tenant_id=user.tenant_id,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                user_info={
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "roles": list(user.roles),
                    "permissions": list(user.permissions)
                }
            )
            
        except Exception as e:
            logger.error("OAuth2 authentication failed", error=str(e))
            raise AuthenticationError("OAuth2 authentication failed")
    
    async def register_user(
        self, 
        request: RegisterRequest,
        created_by: Optional[UUID] = None
    ) -> RegisterResponse:
        """
        Register a new user account.
        
        Args:
            request: User registration request
            created_by: ID of user creating this account
            
        Returns:
            RegisterResponse with user info
        """
        logger.info("Registering new user", email=request.email, tenant_id=request.tenant_id)
        
        try:
            # Check if user already exists
            existing_user = await self.user_repo.get_by_email(request.email)
            if existing_user:
                raise AuthenticationError("User with this email already exists")
            
            # Validate tenant
            tenant = await self.tenant_repo.get_by_id(request.tenant_id)
            if not tenant or not tenant.is_active():
                raise AuthenticationError("Invalid tenant")
            
            # Check tenant user limit
            if not tenant.can_add_user():
                raise AuthenticationError("Tenant user limit exceeded")
            
            # Validate password policy
            if not self._validate_password_policy(request.password):
                raise AuthenticationError("Password does not meet policy requirements")
            
            # Hash password
            password_hash = self.password_service.hash_password(request.password)
            
            # Create user
            user = User(
                tenant_id=request.tenant_id,
                email=request.email,
                username=request.username or request.email.split('@')[0],
                first_name=request.first_name,
                last_name=request.last_name,
                password_hash=password_hash,
                auth_provider=AuthProvider.LOCAL,
                status=UserStatus.PENDING_VERIFICATION,
                created_by=created_by
            )
            
            # Set default role
            if tenant.has_feature("custom_roles") and request.role:
                # TODO: Validate role exists and user has permission to assign it
                pass
            else:
                # Assign default viewer role
                from ...domain.entities.permission import UserRole
                user.add_role(UserRole.VIEWER)
            
            # Save user
            saved_user = await self.user_repo.create(user)
            
            # Update tenant user count
            tenant.increment_user_count()
            await self.tenant_repo.update(tenant)
            
            # Send verification email (implementation depends on email service)
            await self._send_verification_email(saved_user)
            
            logger.info("User registered successfully", user_id=saved_user.id)
            
            return RegisterResponse(
                success=True,
                user_id=saved_user.id,
                message="Account created. Please check your email for verification instructions."
            )
            
        except Exception as e:
            logger.error("User registration failed", error=str(e), email=request.email)
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError("Registration failed")
    
    async def refresh_token(self, request: RefreshTokenRequest) -> LoginResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            request: Refresh token request
            
        Returns:
            LoginResponse with new tokens
        """
        logger.info("Refreshing access token")
        
        try:
            # Validate refresh token
            session = await self.session_repo.get_by_refresh_token(request.refresh_token)
            if not session or not session.is_valid():
                raise AuthenticationError("Invalid refresh token")
            
            # Get user and validate status
            user = await self.user_repo.get_by_id(session.user_id)
            if not user or not user.is_active():
                raise AuthenticationError("User account inactive")
            
            # Generate new access token
            access_token = self._generate_access_token(user)
            expires_at = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
            
            # Update session
            session.extend(self.config.access_token_expire_minutes)
            await self.session_repo.update(session)
            
            logger.info("Token refreshed successfully", user_id=user.id)
            
            return LoginResponse(
                success=True,
                user_id=user.id,
                tenant_id=user.tenant_id,
                access_token=access_token,
                refresh_token=request.refresh_token,  # Keep same refresh token
                expires_at=expires_at.isoformat()
            )
            
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise AuthenticationError("Token refresh failed")
    
    async def logout(self, session_token: str) -> bool:
        """
        Logout user and revoke session.
        
        Args:
            session_token: Session token to revoke
            
        Returns:
            True if logout successful
        """
        logger.info("Logging out user")
        
        try:
            session = await self.session_repo.get_by_token(session_token)
            if session:
                session.revoke(reason="user_logout")
                await self.session_repo.update(session)
                logger.info("Session revoked", session_id=session.id)
            
            return True
            
        except Exception as e:
            logger.error("Logout failed", error=str(e))
            return False
    
    async def setup_mfa(self, user_id: UUID, request: MFASetupRequest) -> Dict[str, any]:
        """
        Set up multi-factor authentication for user.
        
        Args:
            user_id: User ID
            request: MFA setup request
            
        Returns:
            MFA setup information including QR code
        """
        logger.info("Setting up MFA", user_id=user_id)
        
        try:
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Generate MFA secret
            secret = self.mfa_service.generate_secret()
            
            # Generate QR code URL
            qr_code_url = self.mfa_service.get_qr_code_url(
                secret=secret,
                account_name=user.email,
                issuer_name=self.config.mfa_issuer_name
            )
            
            # Generate backup codes
            backup_codes = self.mfa_service.generate_backup_codes()
            
            # Save MFA secret (not enabled yet)
            user.mfa_secret = secret
            user.backup_codes = backup_codes
            await self.user_repo.update(user)
            
            logger.info("MFA setup initiated", user_id=user_id)
            
            return {
                "secret": secret,
                "qr_code_url": qr_code_url,
                "backup_codes": backup_codes,
                "instructions": "Scan the QR code with your authenticator app, then verify with a code to enable MFA."
            }
            
        except Exception as e:
            logger.error("MFA setup failed", error=str(e), user_id=user_id)
            raise AuthenticationError("MFA setup failed")
    
    async def verify_mfa_setup(self, user_id: UUID, request: MFAVerifyRequest) -> bool:
        """
        Verify and enable MFA for user.
        
        Args:
            user_id: User ID
            request: MFA verification request
            
        Returns:
            True if MFA enabled successfully
        """
        logger.info("Verifying MFA setup", user_id=user_id)
        
        try:
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.mfa_secret:
                raise AuthenticationError("MFA not set up")
            
            # Verify code
            if not self.mfa_service.verify_code(user.mfa_secret, request.code):
                raise AuthenticationError("Invalid verification code")
            
            # Enable MFA
            user.mfa_enabled = True
            await self.user_repo.update(user)
            
            logger.info("MFA enabled successfully", user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("MFA verification failed", error=str(e), user_id=user_id)
            raise AuthenticationError("MFA verification failed")
    
    async def disable_mfa(self, user_id: UUID, password: str) -> bool:
        """
        Disable MFA for user (requires password confirmation).
        
        Args:
            user_id: User ID
            password: User password for confirmation
            
        Returns:
            True if MFA disabled successfully
        """
        logger.info("Disabling MFA", user_id=user_id)
        
        try:
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Verify password
            if not self.password_service.verify_password(password, user.password_hash):
                raise AuthenticationError("Invalid password")
            
            # Disable MFA
            user.mfa_enabled = False
            user.mfa_secret = None
            user.backup_codes = []
            await self.user_repo.update(user)
            
            logger.info("MFA disabled successfully", user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("MFA disable failed", error=str(e), user_id=user_id)
            raise AuthenticationError("MFA disable failed")
    
    # Private helper methods
    
    async def _handle_failed_login(self, user: User, ip_address: Optional[str] = None) -> None:
        """Handle failed login attempt."""
        user.increment_failed_login()
        await self.user_repo.update(user)
        
        # Log security event
        logger.warning("Failed login attempt", 
                      user_id=user.id, 
                      attempts=user.failed_login_attempts,
                      ip_address=ip_address)
    
    async def _handle_successful_login(self, user: User, ip_address: Optional[str] = None) -> None:
        """Handle successful login."""
        user.reset_failed_login()
        user.update_last_login(ip_address)
        await self.user_repo.update(user)
        
        logger.info("Successful login", user_id=user.id, ip_address=ip_address)
    
    async def _create_user_session(
        self, 
        user: User, 
        ip_address: Optional[str] = None,
        device_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Create user session and generate tokens."""
        
        # Check concurrent session limit
        active_sessions = await self.session_repo.get_active_sessions(user.id)
        if len(active_sessions) >= self.config.max_concurrent_sessions:
            # Revoke oldest session
            oldest_session = min(active_sessions, key=lambda s: s.last_accessed_at)
            oldest_session.revoke(reason="session_limit_exceeded")
            await self.session_repo.update(oldest_session)
        
        # Generate tokens
        access_token = self._generate_access_token(user)
        refresh_token = self._generate_refresh_token()
        
        # Create session
        expires_at = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        session = UserSession(
            user_id=user.id,
            tenant_id=user.tenant_id,
            session_token=access_token,
            refresh_token=refresh_token,
            ip_address=ip_address or "unknown",
            user_agent=device_info.get("user_agent") if device_info else None,
            device_id=device_info.get("device_id") if device_info else None,
            device_name=device_info.get("device_name") if device_info else None,
            expires_at=expires_at
        )
        
        await self.session_repo.create(session)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": (datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)).isoformat()
        }
    
    def _generate_access_token(self, user: User) -> str:
        """Generate JWT access token."""
        now = datetime.utcnow()
        payload = {
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
            "email": user.email,
            "roles": list(user.roles),
            "permissions": list(user.permissions),
            "iat": now,
            "exp": now + timedelta(minutes=self.config.access_token_expire_minutes),
            "type": "access"
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
    
    def _generate_refresh_token(self) -> str:
        """Generate secure refresh token."""
        return secrets.token_urlsafe(32)
    
    async def _get_or_create_sso_user(
        self,
        email: str,
        first_name: str,
        last_name: str,
        tenant_id: UUID,
        auth_provider: AuthProvider,
        external_id: Optional[str] = None
    ) -> User:
        """Get existing SSO user or create new one."""
        
        # Try to find existing user
        user = await self.user_repo.get_by_email(email)
        
        if user:
            # Update auth provider info if needed
            if user.auth_provider != auth_provider:
                user.auth_provider = auth_provider
                user.external_id = external_id
                await self.user_repo.update(user)
            return user
        
        # Create new SSO user
        user = User(
            tenant_id=tenant_id,
            email=email,
            username=email.split('@')[0],
            first_name=first_name,
            last_name=last_name,
            auth_provider=auth_provider,
            external_id=external_id,
            status=UserStatus.ACTIVE  # SSO users are pre-verified
        )
        
        # Assign default role
        from ...domain.entities.permission import UserRole
        user.add_role(UserRole.VIEWER)
        
        return await self.user_repo.create(user)
    
    def _validate_password_policy(self, password: str) -> bool:
        """Validate password against policy."""
        if len(password) < self.config.min_password_length:
            return False
        
        if self.config.require_password_uppercase and not any(c.isupper() for c in password):
            return False
        
        if self.config.require_password_lowercase and not any(c.islower() for c in password):
            return False
        
        if self.config.require_password_numbers and not any(c.isdigit() for c in password):
            return False
        
        if self.config.require_password_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    async def _send_verification_email(self, user: User) -> None:
        """Send email verification (placeholder for email service integration)."""
        # TODO: Integrate with email service
        logger.info("Verification email sent", user_id=user.id, email=user.email)


class AuthenticationError(Exception):
    """Authentication-related error."""
    pass