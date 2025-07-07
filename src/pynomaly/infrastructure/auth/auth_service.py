"""Enterprise authentication service orchestrating multiple auth providers and MFA."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .enterprise_auth import (
    AuthMethod,
    AuthProvider,
    AuthSession,
    AuthenticationResult,
    LDAPAuthProvider,
    MFAMethod,
    MFAProvider,
    OAuthProvider,
    Permission,
    RBACManager,
    SAMLAuthProvider,
    SessionManager,
    User
)

logger = logging.getLogger(__name__)


class EnterpriseAuthService:
    """Enterprise authentication service with multi-provider support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.auth_providers: Dict[AuthMethod, AuthProvider] = {}
        self.mfa_provider = MFAProvider(config.get("mfa", {}))
        self.rbac_manager = RBACManager()
        self.session_manager = SessionManager(config.get("session", {}))
        
        # Pending MFA sessions
        self.pending_mfa: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize authentication providers."""
        # LDAP provider
        if "ldap" in self.config:
            self.auth_providers[AuthMethod.LDAP] = LDAPAuthProvider(self.config["ldap"])
            logger.info("Initialized LDAP authentication provider")
        
        # SAML provider
        if "saml" in self.config:
            self.auth_providers[AuthMethod.SAML] = SAMLAuthProvider(self.config["saml"])
            logger.info("Initialized SAML authentication provider")
        
        # OAuth provider
        if "oauth" in self.config:
            self.auth_providers[AuthMethod.OAUTH2] = OAuthProvider(self.config["oauth"])
            logger.info("Initialized OAuth authentication provider")
    
    async def authenticate(
        self,
        auth_method: AuthMethod,
        credentials: Dict[str, Any],
        request_info: Optional[Dict[str, Any]] = None
    ) -> AuthenticationResult:
        """Authenticate user with specified method."""
        try:
            # Get authentication provider
            provider = self.auth_providers.get(auth_method)
            if not provider:
                return AuthenticationResult(
                    success=False,
                    error_message=f"Authentication method {auth_method} not supported"
                )
            
            # Perform authentication
            auth_result = await provider.authenticate(credentials, **request_info or {})
            
            if not auth_result.success:
                return auth_result
            
            user = auth_result.user
            if not user:
                return AuthenticationResult(
                    success=False,
                    error_message="Authentication succeeded but no user information"
                )
            
            # Update user with role-based permissions
            await self._update_user_permissions(user)
            
            # Check if MFA is required
            if await self._requires_mfa(user):
                # Store pending authentication
                mfa_token = await self._create_mfa_session(user, auth_method, request_info)
                
                return AuthenticationResult(
                    success=False,
                    requires_mfa=True,
                    mfa_methods=list(user.mfa_methods),
                    next_step="mfa_verification",
                    user=user
                )
            
            # Create session
            session = await self.session_manager.create_session(
                user=user,
                auth_method=auth_method,
                ip_address=request_info.get("ip_address") if request_info else None,
                user_agent=request_info.get("user_agent") if request_info else None
            )
            
            # Update last login
            user.last_login = datetime.now()
            
            return AuthenticationResult(
                success=True,
                user=user,
                session=session
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication failed due to internal error"
            )
    
    async def verify_mfa(
        self,
        mfa_token: str,
        mfa_method: MFAMethod,
        verification_code: str
    ) -> AuthenticationResult:
        """Verify multi-factor authentication."""
        try:
            # Get pending MFA session
            pending = self.pending_mfa.get(mfa_token)
            if not pending:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid or expired MFA token"
                )
            
            user = pending["user"]
            auth_method = pending["auth_method"]
            request_info = pending["request_info"]
            
            # Verify MFA code
            verification_success = False
            
            if mfa_method == MFAMethod.TOTP:
                # Get user's TOTP secret (in real implementation, this would be stored securely)
                totp_secret = user.attributes.get("totp_secret")
                if totp_secret:
                    verification_success = await self.mfa_provider.verify_totp(
                        totp_secret, verification_code
                    )
            
            elif mfa_method in [MFAMethod.SMS, MFAMethod.EMAIL]:
                # Verify against stored code (in real implementation, this would be more secure)
                stored_code = pending.get("verification_code")
                verification_success = (verification_code == stored_code)
            
            if not verification_success:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid verification code"
                )
            
            # Remove pending MFA session
            del self.pending_mfa[mfa_token]
            
            # Create authenticated session
            session = await self.session_manager.create_session(
                user=user,
                auth_method=auth_method,
                ip_address=request_info.get("ip_address") if request_info else None,
                user_agent=request_info.get("user_agent") if request_info else None
            )
            session.mfa_verified = True
            
            # Update last login
            user.last_login = datetime.now()
            
            return AuthenticationResult(
                success=True,
                user=user,
                session=session
            )
            
        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="MFA verification failed"
            )
    
    async def setup_mfa(
        self,
        user_id: str,
        mfa_method: MFAMethod
    ) -> Dict[str, Any]:
        """Setup MFA for user."""
        try:
            if mfa_method == MFAMethod.TOTP:
                # Generate TOTP secret
                secret = await self.mfa_provider.generate_totp_secret(None)
                qr_code_uri = await self.mfa_provider.generate_totp_qr_code(
                    User(user_id=user_id, username=user_id, email="", full_name=""),
                    secret
                )
                
                return {
                    "method": mfa_method.value,
                    "secret": secret,
                    "qr_code_uri": qr_code_uri,
                    "setup_complete": False
                }
            
            elif mfa_method in [MFAMethod.SMS, MFAMethod.EMAIL]:
                return {
                    "method": mfa_method.value,
                    "setup_complete": True
                }
            
            return {
                "error": f"MFA method {mfa_method} not supported"
            }
            
        except Exception as e:
            logger.error(f"MFA setup failed: {e}")
            return {
                "error": "MFA setup failed"
            }
    
    async def verify_session(self, session_id: str) -> Optional[User]:
        """Verify session and return user."""
        try:
            session = await self.session_manager.get_session(session_id)
            if not session:
                return None
            
            # Get user info
            user = await self._get_user_by_id(session.user_id)
            if user:
                # Update user with current permissions
                await self._update_user_permissions(user)
            
            return user
            
        except Exception as e:
            logger.error(f"Session verification failed: {e}")
            return None
    
    async def verify_jwt_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user."""
        try:
            payload = await self.session_manager.verify_jwt_token(token)
            if not payload:
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            user = await self._get_user_by_id(user_id)
            if user:
                await self._update_user_permissions(user)
            
            return user
            
        except Exception as e:
            logger.error(f"JWT token verification failed: {e}")
            return None
    
    async def check_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Check if user has specific permission."""
        return await self.rbac_manager.check_permission(user_id, permission)
    
    async def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        return await self.rbac_manager.assign_role_to_user(user_id, role_id)
    
    async def remove_role(self, user_id: str, role_id: str) -> bool:
        """Remove role from user."""
        return await self.rbac_manager.remove_role_from_user(user_id, role_id)
    
    async def logout(self, session_id: str) -> bool:
        """Logout user session."""
        return await self.session_manager.delete_session(session_id)
    
    async def generate_api_key(self, user_id: str, name: str) -> str:
        """Generate API key for user."""
        # In real implementation, this would be stored securely
        import secrets
        api_key = f"pyn_{secrets.token_urlsafe(32)}"
        
        logger.info(f"Generated API key {name} for user {user_id}")
        return api_key
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        # In real implementation, this would remove from storage
        logger.info(f"Revoked API key {api_key}")
        return True
    
    async def get_user_audit_log(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get user audit log."""
        # In real implementation, this would query audit database
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "login",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "success": True
            }
        ]
    
    async def _update_user_permissions(self, user: User) -> None:
        """Update user with current role-based permissions."""
        user.permissions = await self.rbac_manager.get_user_permissions(user.user_id)
        user.roles = {role.role_id for role in await self.rbac_manager.get_user_roles(user.user_id)}
    
    async def _requires_mfa(self, user: User) -> bool:
        """Check if user requires MFA."""
        # MFA required if user has it enabled or for admin roles
        if user.mfa_enabled and user.mfa_methods:
            return True
        
        # Require MFA for admin users
        if "admin" in user.roles:
            return True
        
        return False
    
    async def _create_mfa_session(
        self,
        user: User,
        auth_method: AuthMethod,
        request_info: Optional[Dict[str, Any]]
    ) -> str:
        """Create MFA session and send verification code."""
        import secrets
        
        mfa_token = secrets.token_urlsafe(32)
        verification_code = await self.mfa_provider.generate_verification_code()
        
        # Store pending MFA session
        self.pending_mfa[mfa_token] = {
            "user": user,
            "auth_method": auth_method,
            "request_info": request_info,
            "verification_code": verification_code,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=5)
        }
        
        # Send verification codes based on user's MFA methods
        if MFAMethod.SMS in user.mfa_methods:
            await self.mfa_provider.send_sms_code(user, verification_code)
        
        if MFAMethod.EMAIL in user.mfa_methods:
            await self.mfa_provider.send_email_code(user, verification_code)
        
        return mfa_token
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID from any provider."""
        # Try each provider to get user info
        for provider in self.auth_providers.values():
            user = await provider.get_user_info(user_id)
            if user:
                return user
        
        # Fallback: create minimal user object
        return User(
            user_id=user_id,
            username=user_id,
            email="",
            full_name=""
        )
    
    async def cleanup_expired_mfa_sessions(self) -> int:
        """Clean up expired MFA sessions."""
        current_time = datetime.now()
        expired_tokens = [
            token for token, session_data in self.pending_mfa.items()
            if session_data["expires_at"] <= current_time
        ]
        
        for token in expired_tokens:
            del self.pending_mfa[token]
        
        logger.info(f"Cleaned up {len(expired_tokens)} expired MFA sessions")
        return len(expired_tokens)


# Authentication middleware for FastAPI
class AuthMiddleware:
    """Authentication middleware for FastAPI applications."""
    
    def __init__(self, auth_service: EnterpriseAuthService):
        self.auth_service = auth_service
    
    async def __call__(self, request, call_next):
        """Process authentication for each request."""
        from fastapi import HTTPException, status
        
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Get token from header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Verify token
        user = await self.auth_service.verify_jwt_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Add user to request state
        request.state.user = user
        
        return await call_next(request)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require authentication)."""
        public_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/auth/login",
            "/auth/oauth",
            "/auth/saml"
        ]
        
        return any(path.startswith(public_path) for public_path in public_paths)


# Permission decorator
def require_permission(permission: Permission):
    """Decorator to require specific permission for endpoint."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            from fastapi import HTTPException, Request, status
            
            # Find request object in args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            # Check if user has permission
            user = getattr(request.state, 'user', None)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if permission not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator