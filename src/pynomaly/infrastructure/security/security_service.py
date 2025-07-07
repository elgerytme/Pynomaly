"""Comprehensive security hardening and audit trail service for anomaly detection platform."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import bcrypt
import jwt

# Optional security libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import pyotp

    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False


class SecurityLevel(Enum):
    """Security levels for different operations."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class AuditEventType(Enum):
    """Types of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    DATA_EXPORT = "data_export"
    ADMIN_ACTION = "admin_action"
    API_CALL = "api_call"


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    # Encryption settings
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval_days: int = 90
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True

    # Authentication settings
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_expiration_minutes: int = 60
    jwt_refresh_expiration_days: int = 7
    enable_2fa: bool = True
    password_min_length: int = 12
    password_require_special_chars: bool = True

    # Authorization settings
    enable_rbac: bool = True
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5
    account_lockout_duration_minutes: int = 15

    # Audit settings
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 365
    enable_integrity_monitoring: bool = True

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20

    # Data protection
    enable_data_masking: bool = True
    enable_pii_detection: bool = True
    data_classification_enabled: bool = True

    # Security monitoring
    enable_anomaly_detection: bool = True
    enable_intrusion_detection: bool = True
    security_scan_interval_hours: int = 24


@dataclass
class AuditEvent:
    """Audit trail event."""

    event_id: UUID
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    security_level: SecurityLevel
    threat_level: Optional[ThreatLevel] = None
    fingerprint: Optional[str] = None

    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for event."""
        data = f"{self.event_type.value}{self.timestamp.isoformat()}{self.user_id}{self.resource}{self.action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class SecurityRole:
    """Security role definition."""

    role_id: str
    name: str
    description: str
    permissions: Set[str]
    security_level: SecurityLevel
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserSession:
    """User session information."""

    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    security_level: SecurityLevel
    mfa_verified: bool = False
    is_active: bool = True


class EncryptionService:
    """Service for data encryption and decryption."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._encryption_keys: Dict[str, bytes] = {}
        self._current_key_id = "default"

        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning(
                "Cryptography library not available. Encryption disabled."
            )
            return

        # Initialize default encryption key
        self._generate_encryption_key(self._current_key_id)

    def _generate_encryption_key(self, key_id: str) -> None:
        """Generate new encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return

        key = Fernet.generate_key()
        self._encryption_keys[key_id] = key
        self.logger.info(f"Generated new encryption key: {key_id}")

    def encrypt_data(
        self, data: Union[str, bytes], key_id: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """Encrypt data and return ciphertext with key ID."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Encryption not available")

        key_id = key_id or self._current_key_id

        if key_id not in self._encryption_keys:
            self._generate_encryption_key(key_id)

        fernet = Fernet(self._encryption_keys[key_id])

        if isinstance(data, str):
            data = data.encode("utf-8")

        encrypted_data = fernet.encrypt(data)
        return encrypted_data, key_id

    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Decryption not available")

        if key_id not in self._encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")

        fernet = Fernet(self._encryption_keys[key_id])
        return fernet.decrypt(encrypted_data)

    def rotate_keys(self) -> str:
        """Rotate encryption keys."""
        new_key_id = f"key_{int(time.time())}"
        self._generate_encryption_key(new_key_id)
        old_key_id = self._current_key_id
        self._current_key_id = new_key_id

        self.logger.info(f"Rotated encryption key from {old_key_id} to {new_key_id}")
        return new_key_id


class AuthenticationService:
    """Service for user authentication and session management."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._active_sessions: Dict[str, UserSession] = {}
        self._failed_login_attempts: Dict[str, List[datetime]] = {}
        self._locked_accounts: Dict[str, datetime] = {}

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")

        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), hashed_password.encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False

    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.config.password_min_length:
            return False

        if self.config.password_require_special_chars:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

            return has_upper and has_lower and has_digit and has_special

        return True

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        totp_code: Optional[str] = None,
    ) -> Optional[UserSession]:
        """Authenticate user and create session."""

        # Check if account is locked
        if username in self._locked_accounts:
            lock_time = self._locked_accounts[username]
            if datetime.utcnow() - lock_time < timedelta(
                minutes=self.config.account_lockout_duration_minutes
            ):
                self.logger.warning(f"Login attempt for locked account: {username}")
                return None
            else:
                # Unlock account
                del self._locked_accounts[username]

        # Simulate user lookup and password verification
        # In practice, this would query a user database
        stored_password_hash = self._get_user_password_hash(username)

        if not stored_password_hash or not self.verify_password(
            password, stored_password_hash
        ):
            self._record_failed_login(username, ip_address)
            return None

        # Check 2FA if enabled
        if self.config.enable_2fa:
            if not totp_code or not self._verify_totp(username, totp_code):
                self.logger.warning(f"2FA verification failed for user: {username}")
                return None

        # Create session
        session = UserSession(
            session_id=str(uuid4()),
            user_id=username,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            security_level=SecurityLevel.INTERNAL,
            mfa_verified=self.config.enable_2fa,
        )

        self._active_sessions[session.session_id] = session

        # Clear failed login attempts
        if username in self._failed_login_attempts:
            del self._failed_login_attempts[username]

        self.logger.info(f"User authenticated successfully: {username}")
        return session

    def generate_jwt_token(self, session: UserSession) -> str:
        """Generate JWT token for session."""
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "security_level": session.security_level.value,
            "mfa_verified": session.mfa_verified,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow()
            + timedelta(minutes=self.config.jwt_expiration_minutes),
        }

        return jwt.encode(payload, self.config.jwt_secret_key, algorithm="HS256")

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token, self.config.jwt_secret_key, algorithms=["HS256"]
            )

            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id not in self._active_sessions:
                return None

            session = self._active_sessions[session_id]

            # Update last activity
            session.last_activity = datetime.utcnow()

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None

    def _record_failed_login(self, username: str, ip_address: str) -> None:
        """Record failed login attempt."""
        if username not in self._failed_login_attempts:
            self._failed_login_attempts[username] = []

        self._failed_login_attempts[username].append(datetime.utcnow())

        # Remove old attempts (older than lockout duration)
        cutoff_time = datetime.utcnow() - timedelta(
            minutes=self.config.account_lockout_duration_minutes
        )
        self._failed_login_attempts[username] = [
            attempt
            for attempt in self._failed_login_attempts[username]
            if attempt > cutoff_time
        ]

        # Check if account should be locked
        if (
            len(self._failed_login_attempts[username])
            >= self.config.max_failed_login_attempts
        ):
            self._locked_accounts[username] = datetime.utcnow()
            self.logger.warning(
                f"Account locked due to failed login attempts: {username}"
            )

    def _get_user_password_hash(self, username: str) -> Optional[str]:
        """Get stored password hash for user (placeholder)."""
        # In practice, this would query a secure user database
        test_users = {
            "admin": self.hash_password("AdminPassword123!"),
            "user": self.hash_password("UserPassword123!"),
        }
        return test_users.get(username)

    def _verify_totp(self, username: str, totp_code: str) -> bool:
        """Verify TOTP code for 2FA."""
        if not PYOTP_AVAILABLE:
            self.logger.warning("TOTP verification not available")
            return True  # Skip 2FA if library not available

        # Get user's TOTP secret (placeholder)
        user_secret = self._get_user_totp_secret(username)
        if not user_secret:
            return False

        totp = pyotp.TOTP(user_secret)
        return totp.verify(totp_code, valid_window=1)

    def _get_user_totp_secret(self, username: str) -> Optional[str]:
        """Get user's TOTP secret (placeholder)."""
        # In practice, this would be stored securely in the database
        return "JBSWY3DPEHPK3PXP"  # Example secret

    def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.is_active = False
            del self._active_sessions[session_id]
            self.logger.info(f"User logged out: {session.user_id}")
            return True
        return False


class AuthorizationService:
    """Service for role-based access control and authorization."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._roles: Dict[str, SecurityRole] = {}
        self._user_roles: Dict[str, Set[str]] = {}

        # Initialize default roles
        self._initialize_default_roles()

    def _initialize_default_roles(self) -> None:
        """Initialize default security roles."""
        roles = [
            SecurityRole(
                role_id="admin",
                name="Administrator",
                description="Full system access",
                permissions={
                    "system:admin",
                    "data:read",
                    "data:write",
                    "data:delete",
                    "model:train",
                    "model:predict",
                    "model:deploy",
                    "user:manage",
                },
                security_level=SecurityLevel.SECRET,
            ),
            SecurityRole(
                role_id="data_scientist",
                name="Data Scientist",
                description="Model development and analysis",
                permissions={
                    "data:read",
                    "model:train",
                    "model:predict",
                    "model:analyze",
                },
                security_level=SecurityLevel.CONFIDENTIAL,
            ),
            SecurityRole(
                role_id="analyst",
                name="Analyst",
                description="Data analysis and reporting",
                permissions={"data:read", "model:predict", "report:generate"},
                security_level=SecurityLevel.RESTRICTED,
            ),
            SecurityRole(
                role_id="viewer",
                name="Viewer",
                description="Read-only access",
                permissions={"data:read", "report:view"},
                security_level=SecurityLevel.INTERNAL,
            ),
        ]

        for role in roles:
            self._roles[role.role_id] = role

    def check_permission(
        self, user_id: str, permission: str, resource_security_level: SecurityLevel
    ) -> bool:
        """Check if user has permission for operation."""
        if not self.config.enable_rbac:
            return True  # RBAC disabled

        user_roles = self._user_roles.get(user_id, set())

        for role_id in user_roles:
            role = self._roles.get(role_id)
            if role:
                # Check permission
                if permission in role.permissions:
                    # Check security level
                    if self._can_access_security_level(
                        role.security_level, resource_security_level
                    ):
                        return True

        self.logger.warning(
            f"Permission denied: user={user_id}, permission={permission}"
        )
        return False

    def _can_access_security_level(
        self, user_level: SecurityLevel, resource_level: SecurityLevel
    ) -> bool:
        """Check if user security level can access resource security level."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.RESTRICTED: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
        }

        return level_hierarchy[user_level] >= level_hierarchy[resource_level]

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        if role_id not in self._roles:
            self.logger.error(f"Role not found: {role_id}")
            return False

        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()

        self._user_roles[user_id].add(role_id)
        self.logger.info(f"Assigned role {role_id} to user {user_id}")
        return True

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user."""
        if user_id in self._user_roles and role_id in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role_id)
            self.logger.info(f"Revoked role {role_id} from user {user_id}")
            return True
        return False


class AuditService:
    """Service for comprehensive audit logging and compliance."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._audit_events: List[AuditEvent] = []
        self._max_events_memory = 10000

        # Audit logger
        self.audit_logger = logging.getLogger("pynomaly.audit")
        if not self.audit_logger.handlers:
            handler = logging.FileHandler("/var/log/pynomaly/audit.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
            self.audit_logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource: str,
        action: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        threat_level: Optional[ThreatLevel] = None,
    ) -> str:
        """Log audit event."""

        if not self.config.enable_audit_logging:
            return ""

        event = AuditEvent(
            event_id=uuid4(),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            security_level=security_level,
            threat_level=threat_level,
        )

        # Store in memory
        self._audit_events.append(event)

        # Limit memory usage
        if len(self._audit_events) > self._max_events_memory:
            self._audit_events = self._audit_events[-self._max_events_memory :]

        # Log to audit file
        self._write_audit_log(event)

        # Check for security violations
        if threat_level and threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._handle_security_violation(event)

        return str(event.event_id)

    def _write_audit_log(self, event: AuditEvent) -> None:
        """Write audit event to log file."""
        log_entry = {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "session_id": event.session_id,
            "source_ip": event.source_ip,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome,
            "security_level": event.security_level.value,
            "threat_level": event.threat_level.value if event.threat_level else None,
            "details": event.details,
            "fingerprint": event.fingerprint,
        }

        self.audit_logger.info(json.dumps(log_entry))

    def _handle_security_violation(self, event: AuditEvent) -> None:
        """Handle security violation events."""
        self.logger.critical(
            f"Security violation detected: {event.event_type.value} by user {event.user_id} "
            f"from {event.source_ip} - {event.action} on {event.resource}"
        )

        # Additional security measures could be triggered here
        # - Block IP address
        # - Lock user account
        # - Send alerts to security team
        # - Trigger incident response

    def search_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Search audit events with filters."""

        filtered_events = self._audit_events

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if threat_level:
            filtered_events = [
                e for e in filtered_events if e.threat_level == threat_level
            ]

        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered_events[:limit]

    def generate_compliance_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified period."""

        events = self.search_audit_events(start_date, end_date, limit=10000)

        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_events": len(events),
            "event_breakdown": {},
            "security_violations": 0,
            "failed_authentications": 0,
            "data_access_events": 0,
            "admin_actions": 0,
            "unique_users": set(),
            "unique_ips": set(),
        }

        for event in events:
            # Event type breakdown
            event_type = event.event_type.value
            if event_type not in report["event_breakdown"]:
                report["event_breakdown"][event_type] = 0
            report["event_breakdown"][event_type] += 1

            # Security metrics
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                report["security_violations"] += 1

            if (
                event.event_type == AuditEventType.AUTHENTICATION
                and event.outcome == "failure"
            ):
                report["failed_authentications"] += 1

            if event.event_type == AuditEventType.DATA_ACCESS:
                report["data_access_events"] += 1

            if event.event_type == AuditEventType.ADMIN_ACTION:
                report["admin_actions"] += 1

            # User and IP tracking
            if event.user_id:
                report["unique_users"].add(event.user_id)

            if event.source_ip:
                report["unique_ips"].add(event.source_ip)

        # Convert sets to counts
        report["unique_users"] = len(report["unique_users"])
        report["unique_ips"] = len(report["unique_ips"])

        return report


class SecurityService:
    """Main security service coordinating all security functions."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security service."""
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.encryption_service = EncryptionService(self.config)
        self.auth_service = AuthenticationService(self.config)
        self.authz_service = AuthorizationService(self.config)
        self.audit_service = AuditService(self.config)

        # Rate limiting
        self._rate_limit_tracking: Dict[str, List[datetime]] = {}

        self.logger.info("Security service initialized")

    def authenticate_request(
        self,
        auth_token: str,
        required_permission: str,
        resource_security_level: SecurityLevel,
        source_ip: str,
        user_agent: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Authenticate and authorize request."""

        # Rate limiting check
        if not self._check_rate_limit(source_ip):
            self.audit_service.log_event(
                AuditEventType.SECURITY_VIOLATION,
                None,
                "rate_limiter",
                "rate_limit_exceeded",
                "blocked",
                {"source_ip": source_ip},
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.MEDIUM,
            )
            return False, None, "Rate limit exceeded"

        # Verify JWT token
        token_payload = self.auth_service.verify_jwt_token(auth_token)
        if not token_payload:
            self.audit_service.log_event(
                AuditEventType.AUTHENTICATION,
                None,
                "auth_token",
                "verify",
                "failure",
                {"reason": "invalid_token"},
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.LOW,
            )
            return False, None, "Invalid authentication token"

        user_id = token_payload.get("user_id")
        session_id = token_payload.get("session_id")

        # Check authorization
        if not self.authz_service.check_permission(
            user_id, required_permission, resource_security_level
        ):
            self.audit_service.log_event(
                AuditEventType.AUTHORIZATION,
                user_id,
                "authorization",
                "check_permission",
                "denied",
                {
                    "permission": required_permission,
                    "security_level": resource_security_level.value,
                },
                session_id=session_id,
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.LOW,
            )
            return False, user_id, "Insufficient permissions"

        # Log successful authorization
        self.audit_service.log_event(
            AuditEventType.AUTHORIZATION,
            user_id,
            "authorization",
            "check_permission",
            "success",
            {
                "permission": required_permission,
                "security_level": resource_security_level.value,
            },
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
        )

        return True, user_id, None

    def _check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        if not self.config.enable_rate_limiting:
            return True

        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=1)

        # Clean old entries
        if identifier in self._rate_limit_tracking:
            self._rate_limit_tracking[identifier] = [
                timestamp
                for timestamp in self._rate_limit_tracking[identifier]
                if timestamp > window_start
            ]
        else:
            self._rate_limit_tracking[identifier] = []

        # Check limit
        if (
            len(self._rate_limit_tracking[identifier])
            >= self.config.rate_limit_requests_per_minute
        ):
            return False

        # Record current request
        self._rate_limit_tracking[identifier].append(current_time)
        return True

    def sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data by removing or masking sensitive information."""
        if not self.config.enable_data_masking:
            return data

        sanitized = data.copy()

        # Define patterns for sensitive data
        sensitive_patterns = {
            "password",
            "secret",
            "key",
            "token",
            "ssn",
            "credit_card",
            "phone",
            "email",
            "address",
            "ip_address",
        }

        for key, value in sanitized.items():
            key_lower = key.lower()

            # Check if key indicates sensitive data
            if any(pattern in key_lower for pattern in sensitive_patterns):
                if isinstance(value, str):
                    sanitized[key] = self._mask_string(value)
                else:
                    sanitized[key] = "[REDACTED]"

        return sanitized

    def _mask_string(self, value: str) -> str:
        """Mask string value for privacy."""
        if len(value) <= 4:
            return "*" * len(value)
        else:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        # Get recent audit events
        recent_events = self.audit_service.search_audit_events(
            start_time=datetime.utcnow() - timedelta(hours=24),
            limit=1000,
        )

        # Calculate security metrics
        failed_auths = len(
            [
                e
                for e in recent_events
                if e.event_type == AuditEventType.AUTHENTICATION
                and e.outcome == "failure"
            ]
        )

        security_violations = len(
            [
                e
                for e in recent_events
                if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
        )

        unique_users = len(set(e.user_id for e in recent_events if e.user_id))
        unique_ips = len(set(e.source_ip for e in recent_events if e.source_ip))

        return {
            "security_status": "healthy" if security_violations == 0 else "at_risk",
            "total_events_24h": len(recent_events),
            "failed_authentications_24h": failed_auths,
            "security_violations_24h": security_violations,
            "unique_users_24h": unique_users,
            "unique_ips_24h": unique_ips,
            "encryption_enabled": CRYPTOGRAPHY_AVAILABLE
            and self.config.enable_encryption_at_rest,
            "2fa_enabled": self.config.enable_2fa,
            "rbac_enabled": self.config.enable_rbac,
            "audit_logging_enabled": self.config.enable_audit_logging,
            "rate_limiting_enabled": self.config.enable_rate_limiting,
        }
