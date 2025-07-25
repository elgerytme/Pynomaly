"""Enterprise authentication management."""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import jwt
import pyotp
import qrcode
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import structlog

from ..config.security_config import SecurityConfig
from ...shared.infrastructure.exceptions.base_exceptions import (
    BaseApplicationError, 
    ErrorCategory, 
    ErrorSeverity
)
from ...shared.infrastructure.logging.structured_logging import StructuredLogger


logger = structlog.get_logger()


class AuthenticationError(BaseApplicationError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthenticationStatus(Enum):
    """Authentication status enumeration."""
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_LOCKED = "account_locked"
    PASSWORD_EXPIRED = "password_expired"
    MFA_REQUIRED = "mfa_required"
    MFA_INVALID = "mfa_invalid"
    SESSION_EXPIRED = "session_expired"
    RATE_LIMITED = "rate_limited"


@dataclass
class UserCredentials:
    """User credential information."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


@dataclass
class AuthenticationResult:
    """Authentication operation result."""
    status: AuthenticationStatus
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    requires_mfa: bool = False
    mfa_qr_code: Optional[str] = None
    message: Optional[str] = None


@dataclass
class SessionInfo:
    """Active session information."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    expires_at: datetime
    is_active: bool = True


class AuthenticationManager:
    """Enterprise authentication management system.
    
    Provides comprehensive authentication capabilities including:
    - Password-based authentication with secure hashing
    - Multi-factor authentication (TOTP)
    - JWT token management
    - Session management
    - Account lockout protection
    - Password policy enforcement
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = StructuredLogger(config.logging)
        
        # Initialize password context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
        
        # Initialize encryption for sensitive data
        self.cipher_suite = Fernet(self._derive_key(config.encryption.at_rest_encryption_key))
        
        # In-memory stores (in production, use external stores)
        self._users: Dict[str, UserCredentials] = {}
        self._sessions: Dict[str, SessionInfo] = {}
        self._rate_limits: Dict[str, List[float]] = {}
    
    def register_user(
        self, 
        username: str, 
        email: str, 
        password: str,
        roles: List[str] = None,
        enable_mfa: bool = True
    ) -> AuthenticationResult:
        """Register a new user with password validation."""
        try:
            # Validate password policy
            if not self._validate_password_policy(password):
                return AuthenticationResult(
                    status=AuthenticationStatus.INVALID_CREDENTIALS,
                    message="Password does not meet policy requirements"
                )
            
            # Check if user already exists
            user_id = self._generate_user_id(username, email)
            if user_id in self._users:
                return AuthenticationResult(
                    status=AuthenticationStatus.INVALID_CREDENTIALS,
                    message="User already exists"
                )
            
            # Generate salt and hash password
            salt = secrets.token_hex(32)
            password_hash = self.pwd_context.hash(password + salt)
            
            # Create user credentials
            user_creds = UserCredentials(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                created_at=datetime.now(timezone.utc),
                password_changed_at=datetime.now(timezone.utc),
                roles=roles or [],
                mfa_enabled=enable_mfa
            )
            
            # Generate MFA secret if enabled
            if enable_mfa:
                user_creds.mfa_secret = pyotp.random_base32()
            
            self._users[user_id] = user_creds
            
            # Generate QR code for MFA setup
            qr_code = None
            if enable_mfa and user_creds.mfa_secret:
                qr_code = self._generate_mfa_qr_code(user_creds)
            
            self.logger.info(
                "User registered successfully",
                user_id=user_id,
                username=username,
                mfa_enabled=enable_mfa
            )
            
            return AuthenticationResult(
                status=AuthenticationStatus.SUCCESS,
                user_id=user_id,
                mfa_qr_code=qr_code,
                message="User registered successfully"
            )
            
        except Exception as e:
            self.logger.error("User registration failed", error=str(e))
            raise AuthenticationError("User registration failed") from e
    
    def authenticate(
        self, 
        username: str, 
        password: str, 
        mfa_code: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown"
    ) -> AuthenticationResult:
        """Authenticate user with password and optional MFA."""
        try:
            # Rate limiting check
            if self._is_rate_limited(ip_address):
                return AuthenticationResult(
                    status=AuthenticationStatus.RATE_LIMITED,
                    message="Too many authentication attempts"
                )
            
            # Find user
            user_creds = self._find_user_by_username(username)
            if not user_creds:
                self._record_failed_attempt(ip_address)
                return AuthenticationResult(
                    status=AuthenticationStatus.INVALID_CREDENTIALS,
                    message="Invalid credentials"
                )
            
            # Check account lockout
            if self._is_account_locked(user_creds):
                return AuthenticationResult(
                    status=AuthenticationStatus.ACCOUNT_LOCKED,
                    message="Account is temporarily locked"
                )
            
            # Verify password
            if not self.pwd_context.verify(password + user_creds.salt, user_creds.password_hash):
                self._record_failed_login(user_creds)
                return AuthenticationResult(
                    status=AuthenticationStatus.INVALID_CREDENTIALS,
                    message="Invalid credentials"
                )
            
            # Check password expiry
            if self._is_password_expired(user_creds):
                return AuthenticationResult(
                    status=AuthenticationStatus.PASSWORD_EXPIRED,
                    message="Password has expired"
                )
            
            # Check MFA requirement
            if user_creds.mfa_enabled:
                if not mfa_code:
                    return AuthenticationResult(
                        status=AuthenticationStatus.MFA_REQUIRED,
                        user_id=user_creds.user_id,
                        message="MFA code required"
                    )
                
                if not self._verify_mfa_code(user_creds, mfa_code):
                    return AuthenticationResult(
                        status=AuthenticationStatus.MFA_INVALID,
                        message="Invalid MFA code"
                    )
            
            # Generate tokens and session
            access_token, refresh_token, expires_at = self._generate_tokens(user_creds)
            session_id = self._create_session(user_creds, ip_address, user_agent)
            
            # Update login info
            user_creds.last_login = datetime.now(timezone.utc)
            user_creds.failed_login_attempts = 0
            user_creds.locked_until = None
            
            self.logger.info(
                "User authenticated successfully",
                user_id=user_creds.user_id,
                username=username,
                session_id=session_id
            )
            
            return AuthenticationResult(
                status=AuthenticationStatus.SUCCESS,
                user_id=user_creds.user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                message="Authentication successful"
            )
            
        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            raise AuthenticationError("Authentication failed") from e
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt.secret_key,
                algorithms=[self.config.jwt.algorithm],
                issuer=self.config.jwt.issuer,
                audience=self.config.jwt.audience
            )
            
            # Check token expiry
            if datetime.fromtimestamp(payload['exp'], timezone.utc) < datetime.now(timezone.utc):
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[AuthenticationResult]:
        """Refresh access token using refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                return None
            
            user_id = payload.get('sub')
            user_creds = self._users.get(user_id)
            if not user_creds:
                return None
            
            # Generate new access token
            access_token, _, expires_at = self._generate_tokens(user_creds)
            
            return AuthenticationResult(
                status=AuthenticationStatus.SUCCESS,
                user_id=user_id,
                access_token=access_token,
                expires_at=expires_at
            )
            
        except Exception:
            return None
    
    def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        try:
            if session_id in self._sessions:
                self._sessions[session_id].is_active = False
                self.logger.info("User logged out", session_id=session_id)
                return True
            return False
        except Exception:
            return False
    
    def change_password(
        self, 
        user_id: str, 
        current_password: str, 
        new_password: str
    ) -> bool:
        """Change user password with validation."""
        try:
            user_creds = self._users.get(user_id)
            if not user_creds:
                return False
            
            # Verify current password
            if not self.pwd_context.verify(current_password + user_creds.salt, user_creds.password_hash):
                return False
            
            # Validate new password policy
            if not self._validate_password_policy(new_password):
                return False
            
            # Update password
            new_salt = secrets.token_hex(32)
            new_password_hash = self.pwd_context.hash(new_password + new_salt)
            
            user_creds.salt = new_salt
            user_creds.password_hash = new_password_hash
            user_creds.password_changed_at = datetime.now(timezone.utc)
            
            self.logger.info("Password changed successfully", user_id=user_id)
            return True
            
        except Exception:
            return False
    
    def enable_mfa(self, user_id: str) -> Optional[str]:
        """Enable MFA for user and return QR code."""
        try:
            user_creds = self._users.get(user_id)
            if not user_creds:
                return None
            
            user_creds.mfa_secret = pyotp.random_base32()
            user_creds.mfa_enabled = True
            
            return self._generate_mfa_qr_code(user_creds)
            
        except Exception:
            return None
    
    # Private helper methods
    
    def _generate_user_id(self, username: str, email: str) -> str:
        """Generate unique user ID."""
        return hashlib.sha256(f"{username}:{email}".encode()).hexdigest()[:16]
    
    def _find_user_by_username(self, username: str) -> Optional[UserCredentials]:
        """Find user by username."""
        for user_creds in self._users.values():
            if user_creds.username == username:
                return user_creds
        return None
    
    def _validate_password_policy(self, password: str) -> bool:
        """Validate password against policy."""
        policy = self.config.policy
        
        if len(password) < policy.password_min_length:
            return False
        
        if policy.password_require_uppercase and not any(c.isupper() for c in password):
            return False
        
        if policy.password_require_lowercase and not any(c.islower() for c in password):
            return False
        
        if policy.password_require_digits and not any(c.isdigit() for c in password):
            return False
        
        if policy.password_require_special and not any(c in "!@#$%^&*()_+-=" for c in password):
            return False
        
        return True
    
    def _is_account_locked(self, user_creds: UserCredentials) -> bool:
        """Check if account is locked."""
        if user_creds.locked_until is None:
            return False
        return datetime.now(timezone.utc) < user_creds.locked_until
    
    def _is_password_expired(self, user_creds: UserCredentials) -> bool:
        """Check if password is expired."""
        if user_creds.password_changed_at is None:
            return True
        
        expire_date = user_creds.password_changed_at + timedelta(
            days=self.config.policy.password_expire_days
        )
        return datetime.now(timezone.utc) > expire_date
    
    def _record_failed_login(self, user_creds: UserCredentials) -> None:
        """Record failed login attempt."""
        user_creds.failed_login_attempts += 1
        
        if user_creds.failed_login_attempts >= self.config.policy.max_login_attempts:
            user_creds.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.policy.lockout_duration_minutes
            )
    
    def _is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited."""
        now = time.time()
        window = 60  # 1 minute window
        max_attempts = self.config.policy.max_login_attempts
        
        if ip_address not in self._rate_limits:
            self._rate_limits[ip_address] = []
        
        # Clean old attempts
        self._rate_limits[ip_address] = [
            t for t in self._rate_limits[ip_address] if now - t < window
        ]
        
        return len(self._rate_limits[ip_address]) >= max_attempts
    
    def _record_failed_attempt(self, ip_address: str) -> None:
        """Record failed attempt for rate limiting."""
        now = time.time()
        if ip_address not in self._rate_limits:
            self._rate_limits[ip_address] = []
        self._rate_limits[ip_address].append(now)
    
    def _verify_mfa_code(self, user_creds: UserCredentials, code: str) -> bool:
        """Verify MFA TOTP code."""
        if not user_creds.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user_creds.mfa_secret)
        return totp.verify(code, valid_window=1)
    
    def _generate_mfa_qr_code(self, user_creds: UserCredentials) -> str:
        """Generate MFA QR code data."""
        if not user_creds.mfa_secret:
            return ""
        
        totp_uri = pyotp.totp.TOTP(user_creds.mfa_secret).provisioning_uri(
            name=user_creds.email,
            issuer_name="Enterprise Security"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Return the URI for client-side QR generation
        return totp_uri
    
    def _generate_tokens(self, user_creds: UserCredentials) -> Tuple[str, str, datetime]:
        """Generate access and refresh tokens."""
        now = datetime.now(timezone.utc)
        access_exp = now + timedelta(minutes=self.config.jwt.access_token_expire_minutes)
        refresh_exp = now + timedelta(days=self.config.jwt.refresh_token_expire_days)
        
        # Access token payload
        access_payload = {
            'sub': user_creds.user_id,
            'username': user_creds.username,
            'email': user_creds.email,
            'roles': user_creds.roles,
            'permissions': user_creds.permissions,
            'type': 'access',
            'iat': now,
            'exp': access_exp,
            'iss': self.config.jwt.issuer,
            'aud': self.config.jwt.audience
        }
        
        # Refresh token payload
        refresh_payload = {
            'sub': user_creds.user_id,
            'type': 'refresh',
            'iat': now,
            'exp': refresh_exp,
            'iss': self.config.jwt.issuer,
            'aud': self.config.jwt.audience
        }
        
        access_token = jwt.encode(
            access_payload,
            self.config.jwt.secret_key,
            algorithm=self.config.jwt.algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.jwt.secret_key,
            algorithm=self.config.jwt.algorithm
        )
        
        return access_token, refresh_token, access_exp
    
    def _create_session(
        self, 
        user_creds: UserCredentials, 
        ip_address: str, 
        user_agent: str
    ) -> str:
        """Create user session."""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        
        session = SessionInfo(
            session_id=session_id,
            user_id=user_creds.user_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=now + timedelta(minutes=self.config.policy.session_timeout_minutes)
        )
        
        self._sessions[session_id] = session
        return session_id
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'enterprise_security_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key