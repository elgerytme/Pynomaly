"""
Advanced Authentication and Authorization System
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from pynomaly.domain.exceptions import SecurityError


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    # JWT Settings
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    refresh_token_expiration_days: int = 7
    
    # Password Settings
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    
    # API Key Settings
    api_key_length: int = 32
    api_key_prefix: str = "pyn_"
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # Session Settings
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    
    # Encryption
    encryption_key: str = Field(default_factory=lambda: Fernet.generate_key().decode())
    
    # Audit Settings
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 90


class User(BaseModel):
    """User model with security fields."""
    
    user_id: str
    username: str
    email: str
    password_hash: str
    api_keys: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None


class Session(BaseModel):
    """User session model."""
    
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class AuditLog(BaseModel):
    """Security audit log entry."""
    
    log_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    action: str
    resource: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    risk_score: int = 0


class SecurityManager:
    """Advanced security manager with comprehensive features."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.cipher_suite = Fernet(config.encryption_key.encode())
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.audit_logs: List[AuditLog] = []
        self.rate_limiter: Dict[str, List[float]] = {}
        self.blocked_ips: set = set()
        self.api_key_registry: Dict[str, str] = {}  # api_key -> user_id
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength against security policy."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate a secure API key for a user."""
        key = self.config.api_key_prefix + secrets.token_urlsafe(self.config.api_key_length)
        self.api_key_registry[key] = user_id
        
        # Add to user's API keys
        if user_id in self.users:
            self.users[user_id].api_keys.append(key)
        
        self.audit_log(
            user_id=user_id,
            action="api_key_generated",
            resource="api_key",
            ip_address="system",
            user_agent="system",
            success=True
        )
        
        return key
    
    def revoke_api_key(self, api_key: str, user_id: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_key_registry:
            del self.api_key_registry[api_key]
            
            # Remove from user's API keys
            if user_id in self.users:
                self.users[user_id].api_keys = [k for k in self.users[user_id].api_keys if k != api_key]
            
            self.audit_log(
                user_id=user_id,
                action="api_key_revoked",
                resource="api_key",
                ip_address="system",
                user_agent="system",
                success=True
            )
            
            return True
        
        return False
    
    def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """Authenticate an API key and return user ID."""
        user_id = self.api_key_registry.get(api_key)
        
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.is_active:
                return user_id
        
        return None
    
    def create_jwt_token(self, user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create a JWT token for a user."""
        payload = {
            "user_id": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token for a user."""
        payload = {
            "user_id": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.config.refresh_token_expiration_days),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def check_rate_limit(self, identifier: str, ip_address: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Check IP-based rate limiting
        if ip_address in self.blocked_ips:
            return False
        
        # Get or create rate limit window
        if identifier not in self.rate_limiter:
            self.rate_limiter[identifier] = []
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute window
        self.rate_limiter[identifier] = [
            t for t in self.rate_limiter[identifier] if t > cutoff_time
        ]
        
        # Check rate limit
        if len(self.rate_limiter[identifier]) >= self.config.rate_limit_requests_per_minute:
            self.audit_log(
                user_id=identifier,
                action="rate_limit_exceeded",
                resource="api",
                ip_address=ip_address,
                user_agent="unknown",
                success=False,
                details={"requests": len(self.rate_limiter[identifier])}
            )
            return False
        
        # Add current request
        self.rate_limiter[identifier].append(current_time)
        return True
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=self.config.session_timeout_minutes)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Clean up old sessions for this user
        self._cleanup_user_sessions(user_id)
        
        self.audit_log(
            user_id=user_id,
            session_id=session_id,
            action="session_created",
            resource="session",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate a session and return session object."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        if datetime.utcnow() > session.expires_at:
            self.invalidate_session(session_id)
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            
            self.audit_log(
                user_id=session.user_id,
                session_id=session_id,
                action="session_invalidated",
                resource="session",
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                success=True
            )
            
            return True
        
        return False
    
    def _cleanup_user_sessions(self, user_id: str) -> None:
        """Clean up old sessions for a user."""
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id and s.is_active]
        
        # Sort by last activity (most recent first)
        user_sessions.sort(key=lambda x: x.last_activity, reverse=True)
        
        # Keep only the most recent sessions
        if len(user_sessions) > self.config.max_concurrent_sessions:
            sessions_to_remove = user_sessions[self.config.max_concurrent_sessions:]
            for session in sessions_to_remove:
                self.invalidate_session(session.session_id)
    
    def audit_log(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        action: str = "",
        resource: str = "",
        ip_address: str = "",
        user_agent: str = "",
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security events for auditing."""
        if not self.config.audit_log_enabled:
            return
        
        log_entry = AuditLog(
            log_id=secrets.token_urlsafe(16),
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
            risk_score=self._calculate_risk_score(action, success, details)
        )
        
        self.audit_logs.append(log_entry)
        
        # Clean up old audit logs
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.audit_log_retention_days)
        self.audit_logs = [log for log in self.audit_logs if log.timestamp > cutoff_date]
    
    def _calculate_risk_score(self, action: str, success: bool, details: Optional[Dict[str, Any]]) -> int:
        """Calculate risk score for security events."""
        score = 0
        
        # Base score for failed actions
        if not success:
            score += 5
        
        # Action-specific scores
        high_risk_actions = ["login_failed", "rate_limit_exceeded", "unauthorized_access"]
        medium_risk_actions = ["password_changed", "api_key_generated", "session_created"]
        
        if action in high_risk_actions:
            score += 8
        elif action in medium_risk_actions:
            score += 3
        
        # Details-based scoring
        if details:
            if "failed_attempts" in details and details["failed_attempts"] > 3:
                score += 10
            if "suspicious_ip" in details and details["suspicious_ip"]:
                score += 7
        
        return min(score, 10)  # Cap at 10
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        recent_logs = [
            log for log in self.audit_logs 
            if log.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        failed_actions = [log for log in recent_logs if not log.success]
        high_risk_events = [log for log in recent_logs if log.risk_score >= 8]
        
        return {
            "total_users": len(self.users),
            "active_sessions": len([s for s in self.sessions.values() if s.is_active]),
            "api_keys_active": len(self.api_key_registry),
            "failed_actions_24h": len(failed_actions),
            "high_risk_events_24h": len(high_risk_events),
            "blocked_ips": len(self.blocked_ips),
            "audit_logs_total": len(self.audit_logs)
        }
    
    def export_audit_logs(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Export audit logs for compliance."""
        filtered_logs = [
            log for log in self.audit_logs
            if start_date <= log.timestamp <= end_date
        ]
        
        return [
            {
                "log_id": log.log_id,
                "user_id": log.user_id,
                "session_id": log.session_id,
                "action": log.action,
                "resource": log.resource,
                "timestamp": log.timestamp.isoformat(),
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "success": log.success,
                "details": log.details,
                "risk_score": log.risk_score
            }
            for log in filtered_logs
        ]


class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    def authenticate_request(self, request_headers: Dict[str, str]) -> Optional[str]:
        """Authenticate a request and return user ID."""
        # Check for API key authentication
        api_key = request_headers.get("X-API-Key") or request_headers.get("Authorization", "").replace("Bearer ", "")
        
        if api_key:
            user_id = self.security_manager.authenticate_api_key(api_key)
            if user_id:
                return user_id
        
        # Check for JWT token authentication
        auth_header = request_headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self.security_manager.verify_jwt_token(token)
            if payload:
                return payload.get("user_id")
        
        return None
    
    def authorize_request(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize a request based on user permissions."""
        if user_id not in self.security_manager.users:
            return False
        
        user = self.security_manager.users[user_id]
        
        # Check if user is active
        if not user.is_active:
            return False
        
        # Check permissions
        required_permission = f"{resource}:{action}"
        return required_permission in user.permissions or "admin" in user.roles
    
    def process_request(self, request_headers: Dict[str, str], ip_address: str, resource: str, action: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Process a request through security middleware."""
        # Check rate limiting
        identifier = ip_address  # Could be user_id if authenticated
        if not self.security_manager.check_rate_limit(identifier, ip_address):
            return False, None, "Rate limit exceeded"
        
        # Authenticate user
        user_id = self.authenticate_request(request_headers)
        if not user_id:
            return False, None, "Authentication required"
        
        # Authorize request
        if not self.authorize_request(user_id, resource, action):
            return False, user_id, "Insufficient permissions"
        
        return True, user_id, None