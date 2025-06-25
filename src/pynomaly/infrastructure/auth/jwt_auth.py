"""JWT authentication implementation for API security.

This module provides JWT-based authentication with:
- Token generation and validation
- User authentication
- Role-based access control
- API key management
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
from pynomaly.infrastructure.config import Settings

logger = logging.getLogger(__name__)


class UserModel(BaseModel):
    """User model for authentication."""

    id: str
    username: str
    email: str
    full_name: str | None = None
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    roles: list[str] = Field(default_factory=list)
    api_keys: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_login: datetime | None = None


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # Subject (user ID)
    exp: datetime  # Expiration
    iat: datetime  # Issued at
    type: str = "access"  # Token type
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None


class JWTAuthService:
    """JWT authentication service."""

    def __init__(self, settings: Settings):
        """Initialize JWT auth service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire = timedelta(seconds=settings.jwt_expiration)
        self.refresh_token_expire = timedelta(days=7)

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # In-memory user store (replace with database in production)
        self._users: dict[str, UserModel] = {}
        self._api_keys: dict[str, str] = {}  # API key -> user ID mapping

        # Initialize default admin user
        self._init_default_users()

    def _init_default_users(self) -> None:
        """Initialize default users for development."""
        if self.settings.app.environment == "development":
            admin_user = UserModel(
                id="admin",
                username="admin",
                email="admin@pynomaly.io",
                full_name="Admin User",
                hashed_password=self.hash_password("admin123"),
                is_superuser=True,
                roles=["admin", "user"],
                api_keys=["dev-api-key-123"],
            )
            self._users[admin_user.id] = admin_user
            self._api_keys["dev-api-key-123"] = admin_user.id

            logger.info("Default admin user created for development")

    def hash_password(self, password: str) -> str:
        """Hash a password.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, user: UserModel) -> TokenResponse:
        """Create access token for user.

        Args:
            user: User model

        Returns:
            Token response
        """
        now = datetime.now(UTC)
        expire = now + self.access_token_expire

        payload = TokenPayload(
            sub=user.id,
            exp=expire,
            iat=now,
            type="access",
            roles=user.roles,
            permissions=self._get_permissions_for_roles(user.roles),
        )

        access_token = jwt.encode(
            payload.model_dump(), self.secret_key, algorithm=self.algorithm
        )

        # Create refresh token
        refresh_expire = now + self.refresh_token_expire
        refresh_payload = payload.model_copy(
            update={"exp": refresh_expire, "type": "refresh"}
        )

        refresh_token = jwt.encode(
            refresh_payload.model_dump(), self.secret_key, algorithm=self.algorithm
        )

        return TokenResponse(
            access_token=access_token,
            expires_in=int(self.access_token_expire.total_seconds()),
            refresh_token=refresh_token,
        )

    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate JWT token.

        Args:
            token: JWT token

        Returns:
            Token payload

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")

    def authenticate_user(self, username: str, password: str) -> UserModel:
        """Authenticate user with username and password.

        Args:
            username: Username or email
            password: Password

        Returns:
            Authenticated user

        Raises:
            AuthenticationError: If authentication fails
        """
        # Find user by username or email
        user = None
        for u in self._users.values():
            if u.username == username or u.email == username:
                user = u
                break

        if not user:
            raise AuthenticationError("Invalid username or password")

        if not self.verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid username or password")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        # Update last login
        user.last_login = datetime.now(UTC)

        return user

    def authenticate_api_key(self, api_key: str) -> UserModel:
        """Authenticate user with API key.

        Args:
            api_key: API key

        Returns:
            Authenticated user

        Raises:
            AuthenticationError: If authentication fails
        """
        user_id = self._api_keys.get(api_key)
        if not user_id:
            raise AuthenticationError("Invalid API key")

        user = self._users.get(user_id)
        if not user:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        return user

    def get_current_user(self, token: str) -> UserModel:
        """Get current user from token.

        Args:
            token: JWT token

        Returns:
            Current user

        Raises:
            AuthenticationError: If user not found
        """
        payload = self.decode_token(token)

        user = self._users.get(payload.sub)
        if not user:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        return user

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New token response

        Raises:
            AuthenticationError: If refresh fails
        """
        payload = self.decode_token(refresh_token)

        if payload.type != "refresh":
            raise AuthenticationError("Invalid token type")

        user = self._users.get(payload.sub)
        if not user:
            raise AuthenticationError("User not found")

        return self.create_access_token(user)

    def check_permissions(
        self, user: UserModel, required_permissions: list[str]
    ) -> bool:
        """Check if user has required permissions.

        Args:
            user: User model
            required_permissions: List of required permissions

        Returns:
            True if user has all permissions
        """
        if user.is_superuser:
            return True

        user_permissions = self._get_permissions_for_roles(user.roles)
        return all(perm in user_permissions for perm in required_permissions)

    def require_permissions(self, user: UserModel, permissions: list[str]) -> None:
        """Require user to have specific permissions.

        Args:
            user: User model
            permissions: Required permissions

        Raises:
            AuthorizationError: If user lacks permissions
        """
        if not self.check_permissions(user, permissions):
            raise AuthorizationError(f"User lacks required permissions: {permissions}")

    def _get_permissions_for_roles(self, roles: list[str]) -> list[str]:
        """Get permissions for given roles.

        Args:
            roles: List of roles

        Returns:
            List of permissions
        """
        # Role -> permissions mapping
        role_permissions = {
            "admin": [
                "detectors:read",
                "detectors:write",
                "detectors:delete",
                "datasets:read",
                "datasets:write",
                "datasets:delete",
                "experiments:read",
                "experiments:write",
                "experiments:delete",
                "users:read",
                "users:write",
                "users:delete",
                "settings:read",
                "settings:write",
            ],
            "user": [
                "detectors:read",
                "detectors:write",
                "datasets:read",
                "datasets:write",
                "experiments:read",
                "experiments:write",
            ],
            "viewer": ["detectors:read", "datasets:read", "experiments:read"],
        }

        permissions = set()
        for role in roles:
            permissions.update(role_permissions.get(role, []))

        return list(permissions)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str | None = None,
        roles: list[str] | None = None,
    ) -> UserModel:
        """Create a new user.

        Args:
            username: Username
            email: Email
            password: Plain text password
            full_name: Full name
            roles: User roles

        Returns:
            Created user

        Raises:
            ValueError: If user already exists
        """
        # Check if user exists
        for user in self._users.values():
            if user.username == username or user.email == email:
                raise ValueError("User already exists")

        user = UserModel(
            id=f"user_{len(self._users) + 1}",
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=self.hash_password(password),
            roles=roles or ["user"],
        )

        self._users[user.id] = user
        logger.info(f"Created user: {username}")

        return user

    def create_api_key(self, user_id: str, key_name: str) -> str:
        """Create API key for user.

        Args:
            user_id: User ID
            key_name: Key name/description

        Returns:
            Generated API key

        Raises:
            ValueError: If user not found
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Generate API key
        import secrets

        api_key = f"pyn_{secrets.token_urlsafe(32)}"

        # Store mapping
        self._api_keys[api_key] = user_id
        user.api_keys.append(api_key)

        logger.info(f"Created API key for user {user_id}: {key_name}")

        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if revoked
        """
        user_id = self._api_keys.pop(api_key, None)
        if user_id:
            user = self._users.get(user_id)
            if user and api_key in user.api_keys:
                user.api_keys.remove(api_key)
            logger.info(f"Revoked API key: {api_key}")
            return True
        return False


# Global auth service instance
_auth_service: JWTAuthService | None = None


def init_auth(settings: Settings) -> JWTAuthService:
    """Initialize global auth service.

    Args:
        settings: Application settings

    Returns:
        Auth service instance
    """
    global _auth_service
    _auth_service = JWTAuthService(settings)
    return _auth_service


def get_auth() -> JWTAuthService | None:
    """Get global auth service instance.

    Returns:
        Auth service or None
    """
    return _auth_service
