"""Production security configuration and hardening for anomaly detection service."""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security configuration levels."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    level: SecurityLevel
    require_https: bool
    api_key_required: bool
    jwt_required: bool
    rate_limit_enabled: bool
    input_validation_strict: bool
    audit_logging_enabled: bool
    encrypt_sensitive_data: bool
    cors_restricted: bool
    csrf_protection: bool
    
    # Rate limiting
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    
    # Authentication
    jwt_expiry_minutes: int
    api_key_length: int
    password_min_length: int
    
    # Data protection
    max_request_size_mb: int
    allowed_file_types: Set[str]
    encrypt_logs: bool
    
    # Network security
    allowed_origins: List[str]
    trusted_proxies: List[str]
    block_suspicious_ips: bool


class ProductionSecurityManager:
    """Production security configuration manager."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        """Initialize security manager."""
        self.security_level = security_level
        self.policy = self._create_security_policy()
        self.blocked_ips: Set[str] = set()
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
    def _create_security_policy(self) -> SecurityPolicy:
        """Create security policy based on security level."""
        if self.security_level == SecurityLevel.DEVELOPMENT:
            return SecurityPolicy(
                level=SecurityLevel.DEVELOPMENT,
                require_https=False,
                api_key_required=False,
                jwt_required=False,
                rate_limit_enabled=False,
                input_validation_strict=False,
                audit_logging_enabled=False,
                encrypt_sensitive_data=False,
                cors_restricted=False,
                csrf_protection=False,
                requests_per_minute=1000,
                requests_per_hour=60000,
                burst_limit=50,
                jwt_expiry_minutes=1440,  # 24 hours
                api_key_length=32,
                password_min_length=8,
                max_request_size_mb=100,
                allowed_file_types={'json', 'csv', 'txt'},
                encrypt_logs=False,
                allowed_origins=['*'],
                trusted_proxies=[],
                block_suspicious_ips=False
            )
        
        elif self.security_level == SecurityLevel.STAGING:
            return SecurityPolicy(
                level=SecurityLevel.STAGING,
                require_https=True,
                api_key_required=True,
                jwt_required=False,
                rate_limit_enabled=True,
                input_validation_strict=True,
                audit_logging_enabled=True,
                encrypt_sensitive_data=True,
                cors_restricted=True,
                csrf_protection=True,
                requests_per_minute=300,
                requests_per_hour=18000,
                burst_limit=20,
                jwt_expiry_minutes=480,  # 8 hours
                api_key_length=48,
                password_min_length=12,
                max_request_size_mb=50,
                allowed_file_types={'json', 'csv'},
                encrypt_logs=True,
                allowed_origins=['https://staging.company.com'],
                trusted_proxies=['10.0.0.0/8', '172.16.0.0/12'],
                block_suspicious_ips=True
            )
        
        elif self.security_level == SecurityLevel.PRODUCTION:
            return SecurityPolicy(
                level=SecurityLevel.PRODUCTION,
                require_https=True,
                api_key_required=True,
                jwt_required=True,
                rate_limit_enabled=True,
                input_validation_strict=True,
                audit_logging_enabled=True,
                encrypt_sensitive_data=True,
                cors_restricted=True,
                csrf_protection=True,
                requests_per_minute=100,
                requests_per_hour=6000,
                burst_limit=10,
                jwt_expiry_minutes=120,  # 2 hours
                api_key_length=64,
                password_min_length=16,
                max_request_size_mb=25,
                allowed_file_types={'json'},
                encrypt_logs=True,
                allowed_origins=['https://app.company.com'],
                trusted_proxies=['10.0.0.0/8'],
                block_suspicious_ips=True
            )
        
        else:  # HIGH_SECURITY
            return SecurityPolicy(
                level=SecurityLevel.HIGH_SECURITY,
                require_https=True,
                api_key_required=True,
                jwt_required=True,
                rate_limit_enabled=True,
                input_validation_strict=True,
                audit_logging_enabled=True,
                encrypt_sensitive_data=True,
                cors_restricted=True,
                csrf_protection=True,
                requests_per_minute=50,
                requests_per_hour=3000,
                burst_limit=5,
                jwt_expiry_minutes=60,  # 1 hour
                api_key_length=128,
                password_min_length=20,
                max_request_size_mb=10,
                allowed_file_types={'json'},
                encrypt_logs=True,
                allowed_origins=[],  # No CORS allowed
                trusted_proxies=[],
                block_suspicious_ips=True
            )
    
    def generate_secure_api_key(self) -> str:
        """Generate a cryptographically secure API key."""
        key_bytes = secrets.token_bytes(self.policy.api_key_length)
        api_key = key_bytes.hex()
        
        logger.info(f"Generated new API key of length {len(api_key)}")
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        salt = os.environ.get('ANOMALY_DETECTION_API_KEY_SALT', 'default_salt')
        combined = f"{api_key}{salt}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def validate_input_size(self, data: Any) -> bool:
        """Validate input data size."""
        if isinstance(data, (list, dict)):
            import json
            size_mb = len(json.dumps(data).encode('utf-8')) / (1024 * 1024)
        elif isinstance(data, str):
            size_mb = len(data.encode('utf-8')) / (1024 * 1024)
        else:
            size_mb = 0
        
        if size_mb > self.policy.max_request_size_mb:
            logger.warning(f"Request size {size_mb:.2f}MB exceeds limit {self.policy.max_request_size_mb}MB")
            return False
        
        return True
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client IP is within rate limits."""
        if not self.policy.rate_limit_enabled:
            return True
        
        now = datetime.utcnow()
        
        # Initialize tracking for new IPs
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        # Clean old attempts
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if now - attempt < timedelta(hours=1)
        ]
        
        # Check hourly limit
        if len(self.failed_attempts[client_ip]) >= self.policy.requests_per_hour:
            logger.warning(f"IP {client_ip} exceeded hourly rate limit")
            return False
        
        # Check minute limit (last 60 seconds)
        recent_attempts = [
            attempt for attempt in self.failed_attempts[client_ip]
            if now - attempt < timedelta(minutes=1)
        ]
        
        if len(recent_attempts) >= self.policy.requests_per_minute:
            logger.warning(f"IP {client_ip} exceeded per-minute rate limit")
            return False
        
        # Record this attempt
        self.failed_attempts[client_ip].append(now)
        return True
    
    def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked."""
        return client_ip in self.blocked_ips
    
    def block_ip(self, client_ip: str, reason: str = "Security violation") -> None:
        """Block an IP address."""
        self.blocked_ips.add(client_ip)
        logger.warning(f"Blocked IP {client_ip}: {reason}")
    
    def validate_file_type(self, filename: str) -> bool:
        """Validate if file type is allowed."""
        if not filename:
            return False
        
        extension = Path(filename).suffix.lower().lstrip('.')
        allowed = extension in self.policy.allowed_file_types
        
        if not allowed:
            logger.warning(f"File type .{extension} not allowed for {filename}")
        
        return allowed
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
        }
        
        if self.policy.require_https:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return headers
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        if not self.policy.cors_restricted:
            return {
                "allow_origins": ["*"],
                "allow_credentials": False,
                "allow_methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["*"]
            }
        
        return {
            "allow_origins": self.policy.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
        }
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage."""
        if not self.policy.encrypt_sensitive_data:
            return data
        
        try:
            from cryptography.fernet import Fernet
            
            # Use environment variable for encryption key
            key = os.environ.get('ANOMALY_DETECTION_ENCRYPTION_KEY')
            if not key:
                # Generate a key if none exists (for development only)
                key = Fernet.generate_key().decode()
                logger.warning("No encryption key found, generated temporary key")
            
            if isinstance(key, str):
                key = key.encode()
            
            f = Fernet(key)
            encrypted = f.encrypt(data.encode())
            return encrypted.decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # In production, this should fail hard
            if self.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.HIGH_SECURITY]:
                raise
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.policy.encrypt_sensitive_data:
            return encrypted_data
        
        try:
            from cryptography.fernet import Fernet
            
            key = os.environ.get('ANOMALY_DETECTION_ENCRYPTION_KEY')
            if not key:
                logger.error("No encryption key found for decryption")
                raise ValueError("Encryption key not available")
            
            if isinstance(key, str):
                key = key.encode()
            
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_data.encode())
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def audit_log(self, event: str, details: Dict[str, Any], client_ip: str = None) -> None:
        """Create audit log entry."""
        if not self.policy.audit_logging_enabled:
            return
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "details": details,
            "client_ip": client_ip,
            "security_level": self.security_level.value
        }
        
        # In production, this should go to a secure audit log system
        logger.info(f"AUDIT: {audit_entry}")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific security configuration."""
        return {
            "security_level": self.security_level.value,
            "policy": {
                "require_https": self.policy.require_https,
                "api_key_required": self.policy.api_key_required,
                "jwt_required": self.policy.jwt_required,
                "rate_limit_enabled": self.policy.rate_limit_enabled,
                "input_validation_strict": self.policy.input_validation_strict,
                "audit_logging_enabled": self.policy.audit_logging_enabled,
                "encrypt_sensitive_data": self.policy.encrypt_sensitive_data,
                "cors_restricted": self.policy.cors_restricted,
                "csrf_protection": self.policy.csrf_protection,
            },
            "limits": {
                "requests_per_minute": self.policy.requests_per_minute,
                "requests_per_hour": self.policy.requests_per_hour,
                "burst_limit": self.policy.burst_limit,
                "max_request_size_mb": self.policy.max_request_size_mb,
            },
            "blocked_ips": len(self.blocked_ips),
            "tracked_ips": len(self.failed_attempts)
        }


# Global security manager instance
_security_manager = None

def get_security_manager() -> ProductionSecurityManager:
    """Get or create the global security manager."""
    global _security_manager
    if _security_manager is None:
        # Determine security level from environment
        security_level_str = os.environ.get('ANOMALY_DETECTION_SECURITY_LEVEL', 'production')
        try:
            security_level = SecurityLevel(security_level_str.lower())
        except ValueError:
            logger.warning(f"Invalid security level '{security_level_str}', defaulting to production")
            security_level = SecurityLevel.PRODUCTION
        
        _security_manager = ProductionSecurityManager(security_level)
        logger.info(f"Initialized security manager with level: {security_level.value}")
    
    return _security_manager


def create_security_middleware():
    """Create security middleware for FastAPI."""
    from fastapi import Request, HTTPException
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    
    class SecurityMiddleware(BaseHTTPMiddleware):
        """Security middleware for request validation and protection."""
        
        def __init__(self, app, security_manager: ProductionSecurityManager = None):
            super().__init__(app)
            self.security_manager = security_manager or get_security_manager()
        
        async def dispatch(self, request: Request, call_next) -> Response:
            """Process request through security checks."""
            client_ip = request.client.host
            
            # Check if IP is blocked
            if self.security_manager.is_ip_blocked(client_ip):
                raise HTTPException(status_code=403, detail="IP address is blocked")
            
            # Check rate limiting
            if not self.security_manager.check_rate_limit(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # HTTPS enforcement
            if self.security_manager.policy.require_https and request.url.scheme != 'https':
                # In production, redirect to HTTPS
                if self.security_manager.security_level == SecurityLevel.PRODUCTION:
                    raise HTTPException(status_code=426, detail="HTTPS required")
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            for header, value in self.security_manager.get_security_headers().items():
                response.headers[header] = value
            
            return response
    
    return SecurityMiddleware


# Environment setup functions
def setup_production_security() -> None:
    """Setup production security configuration."""
    security_manager = get_security_manager()
    
    # Ensure encryption key is set in production
    if security_manager.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.HIGH_SECURITY]:
        if not os.environ.get('ANOMALY_DETECTION_ENCRYPTION_KEY'):
            logger.error("ANOMALY_DETECTION_ENCRYPTION_KEY not set in production environment")
            raise ValueError("Encryption key required for production deployment")
    
    # Validate other required environment variables
    required_vars = [
        'ANOMALY_DETECTION_API_KEY_SALT',
        'ANOMALY_DETECTION_JWT_SECRET_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        if security_manager.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.HIGH_SECURITY]:
            raise ValueError(f"Required environment variables missing: {missing_vars}")
    
    logger.info("Production security configuration validated")


if __name__ == "__main__":
    # Test security configuration
    logging.basicConfig(level=logging.INFO)
    
    # Test different security levels
    for level in SecurityLevel:
        print(f"\n=== {level.value.upper()} Security Configuration ===")
        manager = ProductionSecurityManager(level)
        config = manager.get_environment_config()
        
        print(f"HTTPS Required: {config['policy']['require_https']}")
        print(f"API Key Required: {config['policy']['api_key_required']}")
        print(f"Rate Limit: {config['limits']['requests_per_minute']}/min")
        print(f"Max Request Size: {config['limits']['max_request_size_mb']}MB")
        
        # Test API key generation
        api_key = manager.generate_secure_api_key()
        print(f"Sample API Key Length: {len(api_key)} characters")