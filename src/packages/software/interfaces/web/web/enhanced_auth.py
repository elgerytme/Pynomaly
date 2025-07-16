"""
Enhanced authentication system for Web UI
Supports OAuth2, SAML, MFA, and advanced security features
"""

import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

try:
    from pynomaly_detection.core.config import get_settings
except ImportError:
    # Fallback for testing
    def get_settings():
        from types import SimpleNamespace

        return SimpleNamespace(
            auth=SimpleNamespace(
                secret_key="test-secret-key",
                access_token_expire_minutes=30,
                refresh_token_expire_days=7,
                mfa_enabled=True,
                oauth2_enabled=True,
                saml_enabled=False,
            )
        )


class AuthenticationMethod(Enum):
    """Available authentication methods"""

    PASSWORD = "password"
    OAUTH2_GOOGLE = "oauth2_google"
    OAUTH2_GITHUB = "oauth2_github"
    OAUTH2_MICROSOFT = "oauth2_microsoft"
    SAML = "saml"
    API_KEY = "api_key"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    MFA_EMAIL = "mfa_email"


class UserRole(Enum):
    """User roles for RBAC"""

    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(Enum):
    """Granular permissions"""

    # Dataset permissions
    DATASET_CREATE = "dataset:create"
    DATASET_READ = "dataset:read"
    DATASET_UPDATE = "dataset:update"
    DATASET_DELETE = "dataset:delete"
    DATASET_EXPORT = "dataset:export"

    # Model permissions
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"

    # Analysis permissions
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"
    ANALYSIS_EXPORT = "analysis:export"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"

    # API permissions
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"

    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"


@dataclass
class AuthSession:
    """Enhanced authentication session"""

    session_id: str
    user_id: str
    username: str
    email: str
    roles: list[UserRole]
    permissions: list[Permission]
    authentication_method: AuthenticationMethod
    mfa_verified: bool
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_api_session: bool = False
    session_data: dict[str, Any] | None = None


@dataclass
class OAuth2Provider:
    """OAuth2 provider configuration"""

    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    user_info_url: str
    scopes: list[str]
    enabled: bool = True


@dataclass
class MFAChallenge:
    """Multi-factor authentication challenge"""

    challenge_id: str
    user_id: str
    method: AuthenticationMethod
    code: str
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3


class AuthUser(BaseModel):
    """Enhanced user model for authentication"""

    id: str
    username: str
    email: str
    password_hash: str
    roles: list[UserRole]
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    oauth2_accounts: dict[str, dict[str, Any]] = {}
    failed_login_attempts: int = 0
    last_login: datetime | None = None
    created_at: datetime
    updated_at: datetime


class EnhancedAuthenticationService:
    """Enhanced authentication service with advanced security features"""

    def __init__(self):
        self.settings = get_settings()
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(auto_error=False)

        # In-memory storage (replace with database in production)
        self.users: dict[str, AuthUser] = {}
        self.sessions: dict[str, AuthSession] = {}
        self.mfa_challenges: dict[str, MFAChallenge] = {}
        self.api_keys: dict[str, str] = {}  # api_key -> user_id
        self.refresh_tokens: dict[str, str] = {}  # refresh_token -> user_id

        # OAuth2 providers
        self.oauth2_providers = self._setup_oauth2_providers()

        # Role-based permissions
        self.role_permissions = self._setup_role_permissions()

        # JWT settings
        self.jwt_algorithm = "HS256"
        self.jwt_secret = self.settings.auth.secret_key

    def _setup_oauth2_providers(self) -> dict[str, OAuth2Provider]:
        """Setup OAuth2 providers"""
        providers = {}

        if self.settings.auth.oauth2_enabled:
            # Google OAuth2
            providers["google"] = OAuth2Provider(
                name="Google",
                client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
                authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
                token_url="https://oauth2.googleapis.com/token",
                user_info_url="https://www.googleapis.com/oauth2/v1/userinfo",
                scopes=["openid", "email", "profile"],
                enabled=bool(os.getenv("GOOGLE_CLIENT_ID")),
            )

            # GitHub OAuth2
            providers["github"] = OAuth2Provider(
                name="GitHub",
                client_id=os.getenv("GITHUB_CLIENT_ID", ""),
                client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
                authorize_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                user_info_url="https://api.github.com/user",
                scopes=["user:email"],
                enabled=bool(os.getenv("GITHUB_CLIENT_ID")),
            )

            # Microsoft OAuth2
            providers["microsoft"] = OAuth2Provider(
                name="Microsoft",
                client_id=os.getenv("MICROSOFT_CLIENT_ID", ""),
                client_secret=os.getenv("MICROSOFT_CLIENT_SECRET", ""),
                authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
                user_info_url="https://graph.microsoft.com/v1.0/me",
                scopes=["openid", "email", "profile"],
                enabled=bool(os.getenv("MICROSOFT_CLIENT_ID")),
            )

        return providers

    def _setup_role_permissions(self) -> dict[UserRole, list[Permission]]:
        """Setup role-based permissions mapping"""
        return {
            UserRole.SUPER_ADMIN: [perm for perm in Permission],
            UserRole.ADMIN: [
                Permission.DATASET_CREATE,
                Permission.DATASET_READ,
                Permission.DATASET_UPDATE,
                Permission.DATASET_DELETE,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DELETE,
                Permission.MODEL_TRAIN,
                Permission.MODEL_DEPLOY,
                Permission.ANALYSIS_CREATE,
                Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE,
                Permission.ANALYSIS_DELETE,
                Permission.SYSTEM_MONITOR,
                Permission.SYSTEM_LOGS,
                Permission.USER_CREATE,
                Permission.USER_READ,
                Permission.USER_UPDATE,
                Permission.API_ACCESS,
            ],
            UserRole.DATA_SCIENTIST: [
                Permission.DATASET_CREATE,
                Permission.DATASET_READ,
                Permission.DATASET_UPDATE,
                Permission.DATASET_EXPORT,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_TRAIN,
                Permission.ANALYSIS_CREATE,
                Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE,
                Permission.ANALYSIS_EXPORT,
                Permission.API_ACCESS,
            ],
            UserRole.ANALYST: [
                Permission.DATASET_READ,
                Permission.DATASET_EXPORT,
                Permission.MODEL_READ,
                Permission.ANALYSIS_CREATE,
                Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE,
                Permission.ANALYSIS_EXPORT,
                Permission.API_ACCESS,
            ],
            UserRole.VIEWER: [
                Permission.DATASET_READ,
                Permission.MODEL_READ,
                Permission.ANALYSIS_READ,
            ],
            UserRole.API_USER: [
                Permission.API_ACCESS,
                Permission.DATASET_READ,
                Permission.MODEL_READ,
                Permission.ANALYSIS_READ,
            ],
        }

    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.password_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.password_context.verify(plain_password, hashed_password)

    def create_user(
        self, username: str, email: str, password: str, roles: list[UserRole] = None
    ) -> AuthUser:
        """Create new user with enhanced security"""
        if roles is None:
            roles = [UserRole.VIEWER]

        user_id = str(uuid4())
        password_hash = self.hash_password(password)

        user = AuthUser(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.users[user_id] = user
        return user

    def authenticate_user(self, username: str, password: str) -> AuthUser | None:
        """Authenticate user with username/password"""
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break

        if not user:
            return None

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled"
            )

        # Check failed login attempts
        if user.failed_login_attempts >= 5:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is locked due to too many failed login attempts",
            )

        if not self.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # Reset failed attempts on successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()

        return user

    def create_access_token(
        self,
        user: AuthUser,
        session: AuthSession,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.auth.access_token_expire_minutes
            )

        permissions = []
        for role in user.roles:
            permissions.extend([p.value for p in self.role_permissions.get(role, [])])

        to_encode = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [r.value for r in user.roles],
            "permissions": list(set(permissions)),
            "session_id": session.session_id,
            "mfa_verified": session.mfa_verified,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "pynomaly-web-ui",
            "aud": "pynomaly-api",
        }

        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        refresh_token = secrets.token_urlsafe(32)
        self.refresh_tokens[refresh_token] = user_id
        return refresh_token

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    def create_session(
        self,
        user: AuthUser,
        request: Request,
        authentication_method: AuthenticationMethod,
        mfa_verified: bool = False,
    ) -> AuthSession:
        """Create authentication session"""
        session_id = str(uuid4())
        now = datetime.utcnow()

        permissions = []
        for role in user.roles:
            permissions.extend(self.role_permissions.get(role, []))

        session = AuthSession(
            session_id=session_id,
            user_id=user.id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            permissions=list(set(permissions)),
            authentication_method=authentication_method,
            mfa_verified=mfa_verified,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(hours=8),  # 8-hour session
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> AuthSession | None:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        # Check if session is expired
        if datetime.utcnow() > session.expires_at:
            del self.sessions[session_id]
            return None

        # Update last activity
        session.last_activity = datetime.utcnow()
        return session

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def create_mfa_challenge(
        self, user_id: str, method: AuthenticationMethod
    ) -> MFAChallenge:
        """Create MFA challenge"""
        challenge_id = str(uuid4())

        if method == AuthenticationMethod.MFA_TOTP:
            # For TOTP, the user needs to provide the code from their authenticator app
            code = ""  # Code will be provided by user
        elif method in [AuthenticationMethod.MFA_SMS, AuthenticationMethod.MFA_EMAIL]:
            # Generate 6-digit code for SMS/Email
            code = f"{secrets.randbelow(1000000):06d}"
        else:
            raise ValueError(f"Unsupported MFA method: {method}")

        challenge = MFAChallenge(
            challenge_id=challenge_id,
            user_id=user_id,
            method=method,
            code=code,
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )

        self.mfa_challenges[challenge_id] = challenge
        return challenge

    def verify_mfa_challenge(self, challenge_id: str, provided_code: str) -> bool:
        """Verify MFA challenge"""
        challenge = self.mfa_challenges.get(challenge_id)
        if not challenge:
            return False

        # Check if challenge is expired
        if datetime.utcnow() > challenge.expires_at:
            del self.mfa_challenges[challenge_id]
            return False

        # Increment attempts
        challenge.attempts += 1

        # Check max attempts
        if challenge.attempts > challenge.max_attempts:
            del self.mfa_challenges[challenge_id]
            return False

        # Verify code
        if challenge.method == AuthenticationMethod.MFA_TOTP:
            # For TOTP, verify against user's secret
            user = self.users.get(challenge.user_id)
            if user and user.mfa_secret:
                return self._verify_totp_code(user.mfa_secret, provided_code)
        else:
            # For SMS/Email, compare directly
            if challenge.code == provided_code:
                del self.mfa_challenges[challenge_id]
                return True

        return False

    def _verify_totp_code(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        try:
            import pyotp

            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except ImportError:
            # Fallback implementation without pyotp
            current_time = int(time.time() // 30)
            for time_window in [current_time - 1, current_time, current_time + 1]:
                if self._generate_totp_code(secret, time_window) == code:
                    return True
            return False

    def _generate_totp_code(self, secret: str, time_window: int) -> str:
        """Generate TOTP code for given time window"""
        # Simple TOTP implementation
        key = base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))
        msg = time_window.to_bytes(8, byteorder="big")
        digest = hmac.new(key, msg, hashlib.sha1).digest()

        offset = digest[-1] & 0x0F
        code = (
            (digest[offset] & 0x7F) << 24
            | (digest[offset + 1] & 0xFF) << 16
            | (digest[offset + 2] & 0xFF) << 8
            | (digest[offset + 3] & 0xFF)
        )

        return f"{code % 1000000:06d}"

    def create_api_key(self, user_id: str, name: str) -> str:
        """Create API key for user"""
        api_key = f"pyn_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = user_id
        return api_key

    def verify_api_key(self, api_key: str) -> str | None:
        """Verify API key and return user ID"""
        return self.api_keys.get(api_key)

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False

    def get_oauth2_authorization_url(
        self, provider: str, redirect_uri: str, state: str
    ) -> str:
        """Get OAuth2 authorization URL"""
        if provider not in self.oauth2_providers:
            raise ValueError(f"Unknown OAuth2 provider: {provider}")

        provider_config = self.oauth2_providers[provider]
        if not provider_config.enabled:
            raise ValueError(f"OAuth2 provider {provider} is disabled")

        params = {
            "client_id": provider_config.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(provider_config.scopes),
            "response_type": "code",
            "state": state,
        }

        return f"{provider_config.authorize_url}?{urlencode(params)}"

    def has_permission(
        self, user_or_session: AuthUser | AuthSession, permission: Permission
    ) -> bool:
        """Check if user has specific permission"""
        if isinstance(user_or_session, AuthUser):
            permissions = []
            for role in user_or_session.roles:
                permissions.extend(self.role_permissions.get(role, []))
        else:
            permissions = user_or_session.permissions

        return permission in permissions

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract session from request
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Request object not found",
                    )

                # Get current session
                session = await self.get_current_session(request)
                if not session:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                # Check permission
                if not self.has_permission(session, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission.value}",
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    async def get_current_session(self, request: Request) -> AuthSession | None:
        """Get current session from request"""
        # Try to get token from Authorization header
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            try:
                payload = self.verify_token(token)
                session_id = payload.get("session_id")
                if session_id:
                    return self.get_session(session_id)
            except HTTPException:
                pass

        # Try to get session from cookie
        session_id = request.cookies.get("session_id")
        if session_id:
            return self.get_session(session_id)

        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user_id = self.verify_api_key(api_key)
            if user_id:
                user = self.users.get(user_id)
                if user:
                    # Create temporary session for API key
                    return self.create_session(
                        user,
                        request,
                        AuthenticationMethod.API_KEY,
                        mfa_verified=True,  # API keys bypass MFA
                    )

        return None

    def get_security_metrics(self) -> dict[str, Any]:
        """Get authentication security metrics"""
        now = datetime.utcnow()

        # Active sessions
        active_sessions = [s for s in self.sessions.values() if s.expires_at > now]

        # Failed login attempts
        failed_attempts = sum(u.failed_login_attempts for u in self.users.values())

        # MFA usage
        mfa_enabled_users = sum(1 for u in self.users.values() if u.mfa_enabled)

        return {
            "total_users": len(self.users),
            "active_sessions": len(active_sessions),
            "api_keys": len(self.api_keys),
            "failed_login_attempts": failed_attempts,
            "mfa_enabled_users": mfa_enabled_users,
            "oauth2_providers": len(
                [p for p in self.oauth2_providers.values() if p.enabled]
            ),
            "session_methods": {
                method.value: sum(
                    1 for s in active_sessions if s.authentication_method == method
                )
                for method in AuthenticationMethod
            },
        }


# Global authentication service instance
_auth_service: EnhancedAuthenticationService | None = None


def get_auth_service() -> EnhancedAuthenticationService:
    """Get global authentication service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = EnhancedAuthenticationService()
    return _auth_service
