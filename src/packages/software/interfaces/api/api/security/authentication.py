"""
Advanced authentication and JWT management for Software API.

This module provides secure authentication mechanisms including:
- JWT token generation and validation
- Multi-factor authentication (MFA)
- Session management
- Password hashing and validation
- Account lockout protection
"""

import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any

import bcrypt
import jwt

logger = logging.getLogger(__name__)


class JWTManager:
    """JWT token management with advanced security features."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
        self.refresh_token_expiry = timedelta(days=7)

    def generate_token(
        self, user_id: str, roles: list[str], permissions: list[str]
    ) -> dict[str, Any]:
        """Generate JWT access and refresh tokens."""
        now = datetime.utcnow()

        # Access token payload
        access_payload = {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "iat": now,
            "exp": now + self.token_expiry,
            "jti": secrets.token_hex(16),  # JWT ID for token revocation
            "token_type": "access",
        }

        # Refresh token payload
        refresh_payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "jti": secrets.token_hex(16),
            "token_type": "refresh",
        }

        access_token = jwt.encode(
            access_payload, self.secret_key, algorithm=self.algorithm
        )
        refresh_token = jwt.encode(
            refresh_payload, self.secret_key, algorithm=self.algorithm
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.token_expiry.total_seconds()),
            "expires_at": (now + self.token_expiry).isoformat(),
        }

    def validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("token_type") != "access":
                logger.warning(f"Invalid token type: {payload.get('token_type')}")
                return None

            # Check expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                logger.warning("Token expired")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_token(self, refresh_token: str) -> dict[str, Any] | None:
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(
                refresh_token, self.secret_key, algorithms=[self.algorithm]
            )

            if payload.get("token_type") != "refresh":
                logger.warning("Invalid refresh token type")
                return None

            # Get user information (would typically fetch from database)
            user_id = payload["user_id"]

            # Generate new access token
            return self.generate_token(
                user_id, [], []
            )  # Would fetch actual roles/permissions

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None


class AuthenticationManager:
    """Advanced authentication manager with security features."""

    def __init__(self, jwt_manager: JWTManager):
        self.jwt_manager = jwt_manager
        self.failed_attempts = {}  # user_id -> count
        self.lockout_times = {}  # user_id -> timestamp
        self.max_attempts = 5
        self.lockout_duration = timedelta(minutes=15)

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if user_id not in self.lockout_times:
            return False

        lockout_time = self.lockout_times[user_id]
        if datetime.utcnow() - lockout_time > self.lockout_duration:
            # Lockout period expired
            del self.lockout_times[user_id]
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            return False

        return True

    def record_failed_attempt(self, user_id: str) -> None:
        """Record failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0

        self.failed_attempts[user_id] += 1

        if self.failed_attempts[user_id] >= self.max_attempts:
            self.lockout_times[user_id] = datetime.utcnow()
            logger.warning(f"Account {user_id} locked due to too many failed attempts")

    def reset_failed_attempts(self, user_id: str) -> None:
        """Reset failed attempts counter on successful login."""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.lockout_times:
            del self.lockout_times[user_id]

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        """Authenticate user with advanced security checks."""
        # Check if account is locked
        if self.is_account_locked(username):
            remaining_time = self.lockout_duration - (
                datetime.utcnow() - self.lockout_times[username]
            )
            logger.warning(f"Authentication attempt on locked account: {username}")
            return {
                "error": "account_locked",
                "message": f"Account locked. Try again in {remaining_time.total_seconds():.0f} seconds",
            }

        # Validate password (would typically fetch from database)
        # For demo purposes, using a simple check
        if self._validate_user_credentials(username, password):
            self.reset_failed_attempts(username)

            # Generate JWT tokens
            user_roles = self._get_user_roles(username)
            user_permissions = self._get_user_permissions(username)

            tokens = self.jwt_manager.generate_token(
                username, user_roles, user_permissions
            )

            logger.info(f"User {username} authenticated successfully")
            return {
                "status": "success",
                "user_id": username,
                "tokens": tokens,
                "roles": user_roles,
                "permissions": user_permissions,
            }
        else:
            self.record_failed_attempt(username)
            logger.warning(f"Failed authentication attempt for user: {username}")
            return {
                "error": "invalid_credentials",
                "message": "Invalid username or password",
            }

    def _validate_user_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (placeholder for database lookup)."""
        # In production, this would query the database
        # For demo purposes, accept any non-empty username/password
        return bool(username and password and len(password) >= 8)

    def _get_user_roles(self, username: str) -> list[str]:
        """Get user roles (placeholder for database lookup)."""
        # In production, this would query the database
        return ["user", "analyst"] if username != "admin" else ["admin", "user"]

    def _get_user_permissions(self, username: str) -> list[str]:
        """Get user permissions (placeholder for database lookup)."""
        # In production, this would query the database
        base_permissions = ["read", "write"]
        if username == "admin":
            base_permissions.extend(["delete", "admin"])
        return base_permissions


class MFAManager:
    """Multi-factor authentication manager."""

    def __init__(self):
        self.pending_mfa = {}  # user_id -> {code, expiry}
        self.mfa_code_length = 6
        self.mfa_expiry = timedelta(minutes=5)

    def generate_mfa_code(self, user_id: str) -> str:
        """Generate MFA code for user."""
        code = "".join(
            [str(secrets.randbelow(10)) for _ in range(self.mfa_code_length)]
        )

        self.pending_mfa[user_id] = {
            "code": code,
            "expiry": datetime.utcnow() + self.mfa_expiry,
            "attempts": 0,
        }

        logger.info(f"MFA code generated for user {user_id}")
        return code

    def verify_mfa_code(self, user_id: str, provided_code: str) -> bool:
        """Verify MFA code."""
        if user_id not in self.pending_mfa:
            logger.warning(f"No pending MFA code for user {user_id}")
            return False

        mfa_data = self.pending_mfa[user_id]

        # Check expiry
        if datetime.utcnow() > mfa_data["expiry"]:
            del self.pending_mfa[user_id]
            logger.warning(f"MFA code expired for user {user_id}")
            return False

        # Check attempts
        mfa_data["attempts"] += 1
        if mfa_data["attempts"] > 3:
            del self.pending_mfa[user_id]
            logger.warning(f"Too many MFA attempts for user {user_id}")
            return False

        # Verify code
        if hmac.compare_digest(mfa_data["code"], provided_code):
            del self.pending_mfa[user_id]
            logger.info(f"MFA code verified for user {user_id}")
            return True
        else:
            logger.warning(f"Invalid MFA code for user {user_id}")
            return False

    def cleanup_expired_codes(self) -> None:
        """Clean up expired MFA codes."""
        now = datetime.utcnow()
        expired_users = [
            user_id
            for user_id, data in self.pending_mfa.items()
            if now > data["expiry"]
        ]

        for user_id in expired_users:
            del self.pending_mfa[user_id]

        if expired_users:
            logger.info(f"Cleaned up {len(expired_users)} expired MFA codes")


class SessionManager:
    """Session management with security features."""

    def __init__(self):
        self.active_sessions = {}  # session_id -> session_data
        self.user_sessions = {}  # user_id -> set of session_ids
        self.session_timeout = timedelta(hours=2)

    def create_session(self, user_id: str, request_info: dict[str, Any]) -> str:
        """Create new session."""
        session_id = secrets.token_hex(32)

        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip_address": request_info.get("ip_address"),
            "user_agent": request_info.get("user_agent"),
            "expires_at": datetime.utcnow() + self.session_timeout,
        }

        self.active_sessions[session_id] = session_data

        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)

        logger.info(f"Session created for user {user_id}: {session_id}")
        return session_id

    def validate_session(self, session_id: str) -> dict[str, Any] | None:
        """Validate session and update activity."""
        if session_id not in self.active_sessions:
            return None

        session_data = self.active_sessions[session_id]

        # Check expiry
        if datetime.utcnow() > session_data["expires_at"]:
            self.invalidate_session(session_id)
            return None

        # Update activity
        session_data["last_activity"] = datetime.utcnow()

        return session_data

    def invalidate_session(self, session_id: str) -> None:
        """Invalidate session."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            user_id = session_data["user_id"]

            del self.active_sessions[session_id]

            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

            logger.info(f"Session invalidated: {session_id}")

    def invalidate_all_user_sessions(self, user_id: str) -> None:
        """Invalidate all sessions for a user."""
        if user_id in self.user_sessions:
            session_ids = list(self.user_sessions[user_id])
            for session_id in session_ids:
                self.invalidate_session(session_id)

            logger.info(f"All sessions invalidated for user {user_id}")

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id
            for session_id, data in self.active_sessions.items()
            if now > data["expires_at"]
        ]

        for session_id in expired_sessions:
            self.invalidate_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
