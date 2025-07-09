"""Enhanced JWT authentication implementation with JWKS endpoint and rotating keys.

This module provides JWT-based authentication with:
- Token generation and validation with RSA keys
- JWKS endpoint for public key distribution
- Key rotation capability
- User authentication
- Role-based access control
- API key management
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
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
    kid: str = "1"  # Key ID


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


class JWKSKey(BaseModel):
    """JWKS key representation."""

    kty: str = "RSA"
    use: str = "sig"
    kid: str
    n: str  # Base64url-encoded modulus
    e: str  # Base64url-encoded exponent
    alg: str = "RS256"


class JWKSResponse(BaseModel):
    """JWKS response model."""

    keys: List[JWKSKey]


class EnhancedJWTAuthService:
    """Enhanced JWT authentication service with JWKS endpoint and rotating keys."""

    def __init__(self, settings: Settings):
        """Initialize JWT auth service with rotating keys.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.algorithm = "RS256"  # Use RSA algorithm for JWKS
        self.access_token_expire = timedelta(seconds=settings.jwt_expiration)
        self.refresh_token_expire = timedelta(days=7)

        # Key storage for rotation
        self.current_key_id = "1"
        self.private_keys: Dict[str, rsa.RSAPrivateKey] = {}
        self.public_keys: Dict[str, rsa.RSAPublicKey] = {}
        self.jwks_keys: List[JWKSKey] = []

        # Initialize keys
        self._initialize_keys()

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
            str, list
        ] = {}  # username -> list of attempt timestamps
        self._blacklisted_tokens: set = set()  # For token revocation
        self._password_history: dict[
            str, list
        ] = {}  # user_id -> list of hashed passwords

        # In-memory user store (replace with database in production)
        self._users: dict[str, UserModel] = {}
        self._api_keys: dict[str, str] = {}  # API key -> user ID mapping

        # Initialize default admin user
        self._init_default_users()

    def _initialize_keys(self) -> None:
        """Initialize RSA keys for JWT signing and provide JWKS keys."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        # Store keys
        self.private_keys[self.current_key_id] = private_key
        self.public_keys[self.current_key_id] = public_key

        # Create JWKS representation
        self._update_jwks()

    def _update_jwks(self) -> None:
        """Update JWKS keys from current public keys."""
        self.jwks_keys = []
        
        for key_id, public_key in self.public_keys.items():
            # Get public key numbers
            public_numbers = public_key.public_numbers()
            
            # Convert to base64url format
            n_bytes = public_numbers.n.to_bytes(
                (public_numbers.n.bit_length() + 7) // 8, 'big'
            )
            e_bytes = public_numbers.e.to_bytes(
                (public_numbers.e.bit_length() + 7) // 8, 'big'
            )
            
            n_b64 = base64.urlsafe_b64encode(n_bytes).decode('utf-8').rstrip('=')
            e_b64 = base64.urlsafe_b64encode(e_bytes).decode('utf-8').rstrip('=')

            jwks_key = JWKSKey(
                kid=key_id,
                n=n_b64,
                e=e_b64,
            )
            self.jwks_keys.append(jwks_key)

    def get_jwks(self) -> JWKSResponse:
        """Get JWKS (JSON Web Key Set) for token verification.

        Returns:
            JWKS response with public keys
        """
        return JWKSResponse(keys=self.jwks_keys)

    def rotate_keys(self) -> str:
        """Rotate signing keys.

        Returns:
            New key ID
        """
        # Generate new key ID
        new_key_id = str(int(self.current_key_id) + 1)
        
        # Generate new key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        # Store new keys
        self.private_keys[new_key_id] = private_key
        self.public_keys[new_key_id] = public_key

        # Update current key ID
        self.current_key_id = new_key_id

        # Update JWKS
        self._update_jwks()

        logger.info(f"Keys rotated to new key ID: {new_key_id}")
        return new_key_id

    def cleanup_old_keys(self, keep_count: int = 2) -> None:
        """Clean up old keys, keeping only the most recent ones.

        Args:
            keep_count: Number of keys to keep
        """
        if len(self.private_keys) <= keep_count:
            return

        # Sort keys by ID (assuming sequential)
        sorted_keys = sorted(self.private_keys.keys(), key=int)
        keys_to_remove = sorted_keys[:-keep_count]

        for key_id in keys_to_remove:
            self.private_keys.pop(key_id, None)
            self.public_keys.pop(key_id, None)

        # Update JWKS
        self._update_jwks()

        logger.info(f"Cleaned up {len(keys_to_remove)} old keys")

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
            kid=self.current_key_id,
        )

        # Use current private key for signing
        private_key = self.private_keys[self.current_key_id]
        
        access_token = jwt.encode(
            payload.model_dump(), 
            private_key, 
            algorithm=self.algorithm,
            headers={"kid": self.current_key_id}
        )

        # Create refresh token
        refresh_expire = now + self.refresh_token_expire
        refresh_payload = payload.model_copy(
            update={"exp": refresh_expire, "type": "refresh"}
        )

        refresh_token = jwt.encode(
            refresh_payload.model_dump(), 
            private_key, 
            algorithm=self.algorithm,
            headers={"kid": self.current_key_id}
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
            # Decode header to get key ID
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid or kid not in self.public_keys:
                raise AuthenticationError("Invalid key ID")

            # Use the correct public key for verification
            public_key = self.public_keys[kid]
            
            payload = jwt.decode(
                token, 
                public_key, 
                algorithms=[self.algorithm]
            )
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
                "roles:read",
                "roles:write",
                "roles:delete",
                "permissions:read",
                "permissions:write",
                "permissions:delete",
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

    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist to prevent further use.

        Args:
            token: JWT token to blacklist
        """
        self._blacklisted_tokens.add(token)
        logger.info("Token added to blacklist")

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
_auth_service: EnhancedJWTAuthService | None = None


def init_auth(settings: Settings) -> EnhancedJWTAuthService:
    """Initialize global auth service.

    Args:
        settings: Application settings

    Returns:
        Auth service instance
    """
    global _auth_service
    _auth_service = EnhancedJWTAuthService(settings)
    return _auth_service


def get_auth() -> EnhancedJWTAuthService | None:
    """Get global auth service instance.

    Returns:
        Auth service or None
    """
    return _auth_service
