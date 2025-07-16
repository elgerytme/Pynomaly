"""JWT authentication implementation for API security.

This module provides JWT-based authentication with:
- Token generation and validation
- User authentication
- Role-based access control
- API key management
"""

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


class PasswordRotationStrategy(BaseModel):
    """Password rotation strategy configuration."""

    enabled: bool = True
    max_age_days: int = 90
    force_change_on_first_login: bool = True
    notify_before_expiry_days: int = 7
    password_history_count: int = 12  # Number of previous passwords to remember


class JWTAuthService:
    """Enhanced JWT authentication service with security features."""

    def __init__(self, settings: Settings):
        """Initialize JWT auth service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.secret_key = settings.security.secret_key
        self.algorithm = settings.security.jwt_algorithm
        self.access_token_expire = timedelta(seconds=settings.security.jwt_expiration)
        self.refresh_token_expire = timedelta(days=7)

        # Enhanced password hashing with stronger settings
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12,  # Higher rounds for better security
        )

        # Password rotation configuration
        self.password_rotation = PasswordRotationStrategy()

        # Security tracking
        self._failed_login_attempts: dict[
            str, list[datetime]
        ] = {}  # username -> list of attempt timestamps
        self._blacklisted_tokens: set[str] = set()  # For token revocation
        self._password_history: dict[
            str, list
        ] = {}  # user_id -> list of hashed passwords

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
        # Check if token is blacklisted
        if token in self._blacklisted_tokens:
            raise AuthenticationError("Token has been revoked")

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")

    def _check_account_lockout(self, username: str) -> None:
        """Check if account is locked due to failed login attempts.

        Args:
            username: Username to check

        Raises:
            AuthenticationError: If account is locked
        """
        if username not in self._failed_login_attempts:
            return

        # Clean up old attempts (older than 15 minutes)
        cutoff_time = datetime.now(UTC) - timedelta(minutes=15)
        self._failed_login_attempts[username] = [
            attempt
            for attempt in self._failed_login_attempts[username]
            if attempt > cutoff_time
        ]

        # Check if account should be locked
        if len(self._failed_login_attempts[username]) >= 5:
            raise AuthenticationError(
                "Account temporarily locked due to too many failed login attempts"
            )

    def _record_failed_login(self, username: str) -> None:
        """Record a failed login attempt.

        Args:
            username: Username that failed login
        """
        if username not in self._failed_login_attempts:
            self._failed_login_attempts[username] = []

        self._failed_login_attempts[username].append(datetime.now(UTC))
        logger.warning(f"Failed login attempt for user: {username}")

    def _clear_failed_logins(self, username: str) -> None:
        """Clear failed login attempts for successful login.

        Args:
            username: Username to clear attempts for
        """
        if username in self._failed_login_attempts:
            del self._failed_login_attempts[username]

    def _check_password_rotation(self, user: UserModel) -> bool:
        """Check if user password needs rotation.

        Args:
            user: User model

        Returns:
            True if password needs rotation
        """
        if not self.password_rotation.enabled:
            return False

        # Check if password is too old
        password_age = datetime.now(UTC) - user.created_at
        max_age = timedelta(days=self.password_rotation.max_age_days)

        return password_age > max_age

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
        # Check for account lockout
        self._check_account_lockout(username)

        # Find user by username or email
        user = None
        for u in self._users.values():
            if u.username == username or u.email == username:
                user = u
                break

        if not user:
            self._record_failed_login(username)
            raise AuthenticationError("Invalid username or password")

        if not self.verify_password(password, user.hashed_password):
            self._record_failed_login(username)
            raise AuthenticationError("Invalid username or password")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        # Clear failed login attempts on successful login
        self._clear_failed_logins(username)

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

    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist to prevent further use.

        Args:
            token: JWT token to blacklist
        """
        self._blacklisted_tokens.add(token)
        logger.info("Token added to blacklist")

    def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> bool:
        """Change user password with validation.

        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            True if password changed successfully

        Raises:
            AuthenticationError: If old password is incorrect
            ValueError: If new password is invalid or reused
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Verify old password
        if not self.verify_password(old_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")

        # Check password history to prevent reuse
        if user_id in self._password_history:
            new_password_hash = self.hash_password(new_password)
            for old_hash in self._password_history[user_id]:
                if self.pwd_context.verify(new_password, old_hash):
                    raise ValueError("Cannot reuse a previous password")

        # Update password history
        if user_id not in self._password_history:
            self._password_history[user_id] = []

        self._password_history[user_id].append(user.hashed_password)

        # Keep only the last N passwords
        if (
            len(self._password_history[user_id])
            > self.password_rotation.password_history_count
        ):
            self._password_history[user_id] = self._password_history[user_id][
                -self.password_rotation.password_history_count :
            ]

        # Update user password
        user.hashed_password = self.hash_password(new_password)

        logger.info(f"Password changed for user: {user_id}")
        return True

    def force_password_reset(self, user_id: str) -> None:
        """Force user to reset password on next login.

        Args:
            user_id: User ID to force password reset
        """
        user = self._users.get(user_id)
        if user:
            # Mark password as requiring reset (in production, add a field to user model)
            logger.info(f"Forced password reset for user: {user_id}")

    def rotate_api_key(self, user_id: str, old_api_key: str) -> str:
        """Rotate an API key by replacing it with a new one.

        Args:
            user_id: User ID
            old_api_key: Current API key to replace

        Returns:
            New API key

        Raises:
            ValueError: If API key doesn't belong to user
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError("User not found")

        if old_api_key not in user.api_keys:
            raise ValueError("API key not found for user")

        # Revoke old key
        self.revoke_api_key(old_api_key)

        # Create new key
        new_key = self.create_api_key(user_id, "rotated_key")

        logger.info(f"API key rotated for user: {user_id}")
        return new_key

    def create_password_reset_token(self, email: str) -> str:
        """Create password reset token for user.

        Args:
            email: User email

        Returns:
            Password reset token

        Raises:
            ValueError: If user not found
        """
        # Find user by email
        user = None
        for u in self._users.values():
            if u.email == email:
                user = u
                break

        if not user:
            raise ValueError("User not found")

        # Create reset token (expires in 1 hour)
        now = datetime.now(UTC)
        expire = now + timedelta(hours=1)

        payload = {
            "sub": user.id,
            "email": email,
            "type": "password_reset",
            "exp": expire,
            "iat": now,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Password reset token created for user: {email}")
        return token

    def reset_password_with_token(self, token: str, new_password: str) -> bool:
        """Reset password using reset token.

        Args:
            token: Password reset token
            new_password: New password

        Returns:
            True if password was reset successfully

        Raises:
            ValueError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "password_reset":
                raise ValueError("Invalid token type")

            user_id = payload.get("sub")
            if not user_id:
                raise ValueError("Invalid token")

            user = self._users.get(user_id)
            if not user:
                raise ValueError("User not found")

            # Update password
            user.hashed_password = self.hash_password(new_password)

            logger.info(f"Password reset successfully for user: {user_id}")
            return True

        except jwt.ExpiredSignatureError:
            raise ValueError("Reset token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid reset token")

    def get_user_sessions(self, user_id: str) -> list:
        """Get active sessions for user.

        Args:
            user_id: User ID

        Returns:
            List of active sessions
        """
        # In a real implementation, this would query a session store
        # For now, return mock data
        return [
            {
                "session_id": f"session_{user_id}_1",
                "created_at": datetime.now(UTC).isoformat(),
                "last_activity": datetime.now(UTC).isoformat(),
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "active": True,
            }
        ]

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for user.

        Args:
            user_id: User ID

        Returns:
            Number of sessions invalidated
        """
        # In a real implementation, this would invalidate sessions in session store
        sessions = self.get_user_sessions(user_id)
        count = len(sessions)

        logger.info(f"Invalidated {count} sessions for user: {user_id}")
        return count

    def _invalidate_session(self, session_id: str) -> bool:
        """Invalidate a specific session.

        Args:
            session_id: Session ID

        Returns:
            True if session was invalidated
        """
        # In a real implementation, this would invalidate the session in session store
        logger.info(f"Invalidated session: {session_id}")
        return True

    def create_access_token(
        self, user: UserModel, client_ip: str = None, user_agent: str = None
    ) -> TokenResponse:
        """Create access token for user with enhanced session tracking.

        Args:
            user: User model
            client_ip: Client IP address
            user_agent: User agent string

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

        # Log session creation
        logger.info(f"Session created for user {user.id} from {client_ip}")

        return TokenResponse(
            access_token=access_token,
            expires_in=int(self.access_token_expire.total_seconds()),
            refresh_token=refresh_token,
        )


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
