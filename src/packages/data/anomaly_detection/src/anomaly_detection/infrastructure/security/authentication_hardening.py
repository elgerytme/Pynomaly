"""Advanced authentication hardening with multi-factor authentication and session security."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
import pyotp
import qrcode
from io import BytesIO

from passlib.context import CryptContext
from passlib.hash import bcrypt, scrypt, argon2
import jwt

from ..logging import get_logger

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ..monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    BACKUP_CODES = "backup_codes"


class SessionStatus(Enum):
    """Session status types."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


@dataclass
class AuthenticationAttempt:
    """Authentication attempt record."""
    user_id: str
    method: AuthenticationMethod
    success: bool
    timestamp: datetime
    ip_address: str
    user_agent: str
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    """Session information."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus = SessionStatus.ACTIVE
    mfa_verified: bool = False
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSecurityProfile:
    """User security profile."""
    user_id: str
    password_hash: str
    password_algorithm: str
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    trusted_devices: List[str] = field(default_factory=list)
    security_questions: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class PasswordPolicy:
    """Password policy configuration."""
    
    def __init__(self):
        self.min_length = 12
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_symbols = True
        self.min_symbols = 1
        self.forbidden_patterns = [
            "password", "123456", "qwerty", "admin", "root"
        ]
        self.max_consecutive_chars = 3
        self.password_history_size = 12
        self.max_age_days = 90
        self.min_age_hours = 24
    
    def validate_password(self, password: str, user_history: List[str] = None) -> Tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")
        if len(password) > self.max_length:
            errors.append(f"Password must be no more than {self.max_length} characters")
        
        # Character requirements
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.require_symbols:
            symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            symbol_count = sum(1 for c in password if c in symbols)
            if symbol_count < self.min_symbols:
                errors.append(f"Password must contain at least {self.min_symbols} symbol(s)")
        
        # Forbidden patterns
        password_lower = password.lower()
        for pattern in self.forbidden_patterns:
            if pattern in password_lower:
                errors.append(f"Password cannot contain '{pattern}'")
        
        # Consecutive characters
        consecutive_count = 1
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                consecutive_count += 1
                if consecutive_count > self.max_consecutive_chars:
                    errors.append(f"Password cannot have more than {self.max_consecutive_chars} consecutive identical characters")
                    break
            else:
                consecutive_count = 1
        
        # History check
        if user_history:
            pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
            for old_hash in user_history[-self.password_history_size:]:
                if pwd_context.verify(password, old_hash):
                    errors.append("Password has been used recently")
                    break
        
        return len(errors) == 0, errors


class AdvancedPasswordHasher:
    """Advanced password hashing with multiple algorithms."""
    
    def __init__(self):
        # Primary context with modern algorithms
        self.primary_context = CryptContext(
            schemes=["argon2", "scrypt", "bcrypt"],
            default="argon2",
            deprecated="auto",
            
            # Argon2 configuration
            argon2__memory_cost=65536,  # 64 MB
            argon2__time_cost=3,        # 3 iterations
            argon2__parallelism=4,      # 4 threads
            
            # Scrypt configuration
            scrypt__rounds=32768,       # N parameter
            scrypt__block_size=8,       # r parameter
            scrypt__parallelism=1,      # p parameter
            
            # Bcrypt configuration
            bcrypt__rounds=12,          # Cost factor
        )
        
        # Legacy context for migration
        self.legacy_context = CryptContext(
            schemes=["argon2", "scrypt", "bcrypt", "pbkdf2_sha256"],
            deprecated="auto"
        )
    
    def hash_password(self, password: str, algorithm: str = "argon2") -> Tuple[str, str]:
        """Hash password with specified algorithm."""
        if algorithm not in ["argon2", "scrypt", "bcrypt"]:
            algorithm = "argon2"
        
        # Add pepper (application-level secret)
        peppered_password = self._add_pepper(password)
        
        # Hash with selected algorithm
        password_hash = self.primary_context.hash(peppered_password, scheme=algorithm)
        
        return password_hash, algorithm
    
    def verify_password(self, password: str, password_hash: str, algorithm: str = None) -> bool:
        """Verify password against hash."""
        try:
            # Add pepper
            peppered_password = self._add_pepper(password)
            
            # Try primary context first
            if self.primary_context.verify(peppered_password, password_hash):
                return True
            
            # Fall back to legacy context for old hashes
            return self.legacy_context.verify(peppered_password, password_hash)
            
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def needs_rehash(self, password_hash: str) -> bool:
        """Check if password hash needs to be updated."""
        return self.primary_context.needs_update(password_hash)
    
    def _add_pepper(self, password: str) -> str:
        """Add application-level pepper to password."""
        # In production, this should come from environment/config
        pepper = "your-application-pepper-key-here"
        return password + pepper


class MultiFactorAuthenticator:
    """Multi-factor authentication manager."""
    
    def __init__(self):
        self.backup_code_length = 8
        self.backup_code_count = 10
        self.totp_window = 1  # Allow 1 step tolerance
        self.sms_code_length = 6
        self.sms_expiry_minutes = 5
    
    def setup_totp(self, user_id: str, issuer: str = "Anomaly Detection") -> Tuple[str, str]:
        """Set up TOTP (Time-based One-Time Password) for user."""
        # Generate secret
        secret = pyotp.random_base32()
        
        # Create TOTP instance
        totp = pyotp.TOTP(secret)
        
        # Generate provisioning URI
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        return secret, provisioning_uri
    
    def generate_qr_code(self, provisioning_uri: str) -> bytes:
        """Generate QR code for TOTP setup."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return img_buffer.getvalue()
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=self.totp_window)
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False
    
    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes for account recovery."""
        codes = []
        for _ in range(self.backup_code_count):
            code = secrets.token_hex(self.backup_code_length // 2).upper()
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        
        return codes
    
    def hash_backup_codes(self, codes: List[str]) -> List[str]:
        """Hash backup codes for secure storage."""
        hasher = AdvancedPasswordHasher()
        hashed_codes = []
        
        for code in codes:
            hashed_code, _ = hasher.hash_password(code.replace("-", ""))
            hashed_codes.append(hashed_code)
        
        return hashed_codes
    
    def verify_backup_code(self, code: str, hashed_codes: List[str]) -> Tuple[bool, int]:
        """Verify backup code and return index if valid."""
        clean_code = code.replace("-", "").upper()
        hasher = AdvancedPasswordHasher()
        
        for i, hashed_code in enumerate(hashed_codes):
            if hasher.verify_password(clean_code, hashed_code):
                return True, i
        
        return False, -1


class SessionManager:
    """Advanced session management with security features."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self.session_timeout = timedelta(hours=24)
        self.idle_timeout = timedelta(minutes=30)
        self.max_sessions_per_user = 5
        self.suspicious_activity_threshold = 3  # Different IPs in short time
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        mfa_verified: bool = False,
        permissions: List[str] = None
    ) -> str:
        """Create new session with security checks."""
        # Generate secure session ID
        session_id = self._generate_session_id()
        
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        # Check session limits
        if user_id in self.user_sessions:
            if len(self.user_sessions[user_id]) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_session_id = self.user_sessions[user_id][0]
                self.revoke_session(oldest_session_id)
        
        # Create session
        now = datetime.utcnow()
        session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified,
            permissions=permissions or []
        )
        
        # Store session
        self.sessions[session_id] = session
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Check for suspicious activity
        self._check_suspicious_activity(user_id, ip_address)
        
        logger.info(f"Session created for user {user_id}", session_id=session_id)
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Tuple[bool, Optional[SessionInfo]]:
        """Validate session with security checks."""
        if session_id not in self.sessions:
            return False, None
        
        session = self.sessions[session_id]
        now = datetime.utcnow()
        
        # Check status
        if session.status != SessionStatus.ACTIVE:
            return False, None
        
        # Check session timeout
        if now - session.created_at > self.session_timeout:
            session.status = SessionStatus.EXPIRED
            return False, None
        
        # Check idle timeout
        if now - session.last_activity > self.idle_timeout:
            session.status = SessionStatus.EXPIRED
            return False, None
        
        # Check IP consistency (optional security measure)
        if ip_address and session.ip_address != ip_address:
            # Log suspicious activity but don't immediately invalidate
            logger.warning(f"IP mismatch for session {session_id}: {session.ip_address} vs {ip_address}")
            session.status = SessionStatus.SUSPICIOUS
            
            # Could implement IP validation policy here
            # For now, just log and continue
        
        # Update last activity
        session.last_activity = now
        
        return True, session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.REVOKED
        
        # Remove from user sessions
        if session.user_id in self.user_sessions:
            try:
                self.user_sessions[session.user_id].remove(session_id)
            except ValueError:
                pass
        
        logger.info(f"Session revoked", session_id=session_id)
        
        return True
    
    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        if user_id not in self.user_sessions:
            return 0
        
        session_ids = self.user_sessions[user_id].copy()
        revoked_count = 0
        
        for session_id in session_ids:
            if self.revoke_session(session_id):
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        
        return revoked_count
    
    def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        if user_id not in self.user_sessions:
            return []
        
        sessions = []
        for session_id in self.user_sessions[user_id]:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.status == SessionStatus.ACTIVE:
                    sessions.append(session)
        
        return sessions
    
    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_urlsafe(32)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (session.status != SessionStatus.ACTIVE or
                now - session.created_at > self.session_timeout or
                now - session.last_activity > self.idle_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)
    
    def _check_suspicious_activity(self, user_id: str, ip_address: str):
        """Check for suspicious session activity."""
        user_sessions = self.get_user_sessions(user_id)
        
        # Check for multiple IPs in short time
        recent_sessions = [
            s for s in user_sessions
            if datetime.utcnow() - s.created_at < timedelta(minutes=10)
        ]
        
        unique_ips = set(s.ip_address for s in recent_sessions)
        if len(unique_ips) >= self.suspicious_activity_threshold:
            logger.warning(f"Suspicious activity: {len(unique_ips)} IPs for user {user_id}")
            
            # Mark sessions as suspicious
            for session in recent_sessions:
                if session.ip_address != ip_address:  # Don't mark current session
                    session.status = SessionStatus.SUSPICIOUS


class AuthenticationHardening:
    """Main authentication hardening orchestrator."""
    
    def __init__(self):
        self.password_policy = PasswordPolicy()
        self.password_hasher = AdvancedPasswordHasher()
        self.mfa_authenticator = MultiFactorAuthenticator()
        self.session_manager = SessionManager()
        
        # User profiles storage (in production, this would be a database)
        self.user_profiles: Dict[str, UserSecurityProfile] = {}
        
        # Authentication attempts tracking
        self.auth_attempts: List[AuthenticationAttempt] = []
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.password_reset_expiry = timedelta(hours=1)
        
        logger.info("Authentication hardening initialized")
    
    def create_user(
        self,
        user_id: str,
        password: str,
        require_mfa: bool = False
    ) -> Tuple[bool, List[str]]:
        """Create new user with security validation."""
        # Validate password
        valid, errors = self.password_policy.validate_password(password)
        if not valid:
            return False, errors
        
        # Hash password
        password_hash, algorithm = self.password_hasher.hash_password(password)
        
        # Create security profile
        profile = UserSecurityProfile(
            user_id=user_id,
            password_hash=password_hash,
            password_algorithm=algorithm,
            mfa_enabled=require_mfa
        )
        
        # Set up MFA if required
        if require_mfa:
            secret, _ = self.mfa_authenticator.setup_totp(user_id)
            profile.mfa_secret = secret
            profile.backup_codes = self.mfa_authenticator.hash_backup_codes(
                self.mfa_authenticator.generate_backup_codes()
            )
        
        self.user_profiles[user_id] = profile
        
        logger.info(f"User created", user_id=user_id, mfa_enabled=require_mfa)
        
        return True, []
    
    async def authenticate_user(
        self,
        user_id: str,
        password: str,
        mfa_token: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown"
    ) -> Tuple[bool, Optional[str], List[str]]:
        """Authenticate user with comprehensive security checks."""
        attempt = AuthenticationAttempt(
            user_id=user_id,
            method=AuthenticationMethod.PASSWORD,
            success=False,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        try:
            # Check if user exists
            if user_id not in self.user_profiles:
                attempt.failure_reason = "User not found"
                self.auth_attempts.append(attempt)
                return False, None, ["Invalid credentials"]
            
            profile = self.user_profiles[user_id]
            
            # Check if account is locked
            if self._is_user_locked(profile):
                attempt.failure_reason = "Account locked"
                self.auth_attempts.append(attempt)
                return False, None, ["Account is temporarily locked"]
            
            # Verify password
            if not self.password_hasher.verify_password(password, profile.password_hash):
                # Record failed attempt
                profile.failed_attempts += 1
                
                # Lock account if too many failures
                if profile.failed_attempts >= self.max_failed_attempts:
                    profile.locked_until = datetime.utcnow() + self.lockout_duration
                    logger.warning(f"Account locked due to failed attempts", user_id=user_id)
                
                attempt.failure_reason = "Invalid password"
                self.auth_attempts.append(attempt)
                return False, None, ["Invalid credentials"]
            
            # Check if password needs rehashing
            if self.password_hasher.needs_rehash(profile.password_hash):
                new_hash, algorithm = self.password_hasher.hash_password(password)
                profile.password_hash = new_hash
                profile.password_algorithm = algorithm
                logger.info(f"Password rehashed for user {user_id}")
            
            # Reset failed attempts on successful password
            profile.failed_attempts = 0
            profile.locked_until = None
            
            # Check MFA if enabled
            mfa_verified = True
            if profile.mfa_enabled:
                if not mfa_token:
                    return False, None, ["MFA token required"]
                
                # Verify TOTP
                if profile.mfa_secret:
                    mfa_verified = self.mfa_authenticator.verify_totp(profile.mfa_secret, mfa_token)
                
                # Try backup codes if TOTP fails
                if not mfa_verified and profile.backup_codes:
                    mfa_verified, code_index = self.mfa_authenticator.verify_backup_code(
                        mfa_token, profile.backup_codes
                    )
                    
                    # Remove used backup code
                    if mfa_verified and code_index >= 0:
                        profile.backup_codes.pop(code_index)
                        logger.info(f"Backup code used for user {user_id}")
                
                if not mfa_verified:
                    attempt.failure_reason = "Invalid MFA token"
                    self.auth_attempts.append(attempt)
                    return False, None, ["Invalid MFA token"]
            
            # Create session
            session_id = self.session_manager.create_session(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                mfa_verified=mfa_verified,
                permissions=[]  # Would be loaded from user roles
            )
            
            # Record successful attempt
            attempt.success = True
            attempt.metadata = {"session_id": session_id, "mfa_verified": mfa_verified}
            self.auth_attempts.append(attempt)
            
            # Update profile
            profile.updated_at = datetime.utcnow()
            
            logger.info(f"User authenticated successfully", user_id=user_id, session_id=session_id)
            
            return True, session_id, []
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            attempt.failure_reason = "Authentication error"
            self.auth_attempts.append(attempt)
            return False, None, ["Authentication failed"]
    
    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> Tuple[bool, List[str]]:
        """Change user password with security validation."""
        if user_id not in self.user_profiles:
            return False, ["User not found"]
        
        profile = self.user_profiles[user_id]
        
        # Verify old password
        if not self.password_hasher.verify_password(old_password, profile.password_hash):
            return False, ["Current password is incorrect"]
        
        # Validate new password
        # Get password history (would be from database in production)
        password_history = []  # Should load from database
        
        valid, errors = self.password_policy.validate_password(new_password, password_history)
        if not valid:
            return False, errors
        
        # Check minimum age
        if (profile.last_password_change and 
            datetime.utcnow() - profile.last_password_change < 
            timedelta(hours=self.password_policy.min_age_hours)):
            return False, ["Password was changed too recently"]
        
        # Hash new password
        new_hash, algorithm = self.password_hasher.hash_password(new_password)
        
        # Update profile
        profile.password_hash = new_hash
        profile.password_algorithm = algorithm
        profile.last_password_change = datetime.utcnow()
        profile.updated_at = datetime.utcnow()
        
        # Revoke all existing sessions to force re-authentication
        self.session_manager.revoke_all_user_sessions(user_id)
        
        logger.info(f"Password changed for user {user_id}")
        
        return True, []
    
    def enable_mfa(self, user_id: str) -> Tuple[bool, str, bytes, List[str]]:
        """Enable MFA for user."""
        if user_id not in self.user_profiles:
            return False, "", b"", ["User not found"]
        
        profile = self.user_profiles[user_id]
        
        # Set up TOTP
        secret, provisioning_uri = self.mfa_authenticator.setup_totp(user_id)
        qr_code = self.mfa_authenticator.generate_qr_code(provisioning_uri)
        
        # Generate backup codes
        backup_codes = self.mfa_authenticator.generate_backup_codes()
        hashed_codes = self.mfa_authenticator.hash_backup_codes(backup_codes)
        
        # Update profile
        profile.mfa_enabled = True
        profile.mfa_secret = secret
        profile.backup_codes = hashed_codes
        profile.updated_at = datetime.utcnow()
        
        logger.info(f"MFA enabled for user {user_id}")
        
        return True, provisioning_uri, qr_code, backup_codes
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Tuple[bool, Optional[SessionInfo]]:
        """Validate session with security checks."""
        return self.session_manager.validate_session(session_id, ip_address)
    
    def get_security_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive security status for user."""
        if user_id not in self.user_profiles:
            return {"error": "User not found"}
        
        profile = self.user_profiles[user_id]
        sessions = self.session_manager.get_user_sessions(user_id)
        
        # Get recent authentication attempts
        recent_attempts = [
            attempt for attempt in self.auth_attempts
            if (attempt.user_id == user_id and 
                datetime.utcnow() - attempt.timestamp < timedelta(days=7))
        ]
        
        return {
            "user_id": user_id,
            "account_status": {
                "locked": self._is_user_locked(profile),
                "locked_until": profile.locked_until.isoformat() if profile.locked_until else None,
                "failed_attempts": profile.failed_attempts,
                "mfa_enabled": profile.mfa_enabled,
                "last_password_change": profile.last_password_change.isoformat() if profile.last_password_change else None,
                "password_age_days": (datetime.utcnow() - profile.last_password_change).days if profile.last_password_change else None
            },
            "sessions": {
                "active_count": len(sessions),
                "max_allowed": self.session_manager.max_sessions_per_user,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "created_at": s.created_at.isoformat(),
                        "last_activity": s.last_activity.isoformat(),
                        "ip_address": s.ip_address,
                        "status": s.status.value,
                        "mfa_verified": s.mfa_verified
                    }
                    for s in sessions
                ]
            },
            "recent_activity": {
                "total_attempts": len(recent_attempts),
                "successful_attempts": len([a for a in recent_attempts if a.success]),
                "failed_attempts": len([a for a in recent_attempts if not a.success]),
                "unique_ips": len(set(a.ip_address for a in recent_attempts)),
                "last_successful": max(
                    (a.timestamp for a in recent_attempts if a.success),
                    default=None
                )
            }
        }
    
    def _is_user_locked(self, profile: UserSecurityProfile) -> bool:
        """Check if user account is locked."""
        if not profile.locked_until:
            return False
        
        return datetime.utcnow() < profile.locked_until


# Global authentication hardening instance
_auth_hardening: Optional[AuthenticationHardening] = None


def get_authentication_hardening() -> AuthenticationHardening:
    """Get the global authentication hardening instance."""
    global _auth_hardening
    
    if _auth_hardening is None:
        _auth_hardening = AuthenticationHardening()
    
    return _auth_hardening


def initialize_authentication_hardening() -> AuthenticationHardening:
    """Initialize the global authentication hardening."""
    global _auth_hardening
    _auth_hardening = AuthenticationHardening()
    return _auth_hardening