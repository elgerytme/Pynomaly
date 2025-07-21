"""
Security Configuration for Pynomaly Detection
=============================================

Comprehensive security configuration providing:
- Production-ready security settings
- Environment-based configuration
- Secret management
- Security policy enforcement
- Audit logging configuration
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import secrets
import hashlib

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Environment
    environment: Environment = Environment.PRODUCTION
    
    # Debug settings - NEVER enable in production
    debug_enabled: bool = False
    debug_logging: bool = False
    
    # Authentication
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    session_timeout_minutes: int = 30
    
    # API Security
    api_rate_limit_per_minute: int = 100
    api_rate_limit_burst: int = 200
    cors_allowed_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = False
    max_request_size_mb: int = 10
    
    # Database Security
    db_connection_timeout: int = 30
    db_max_connections: int = 20
    db_ssl_required: bool = True
    db_encrypt_sensitive_data: bool = True
    
    # Logging Security
    log_level: LogLevel = LogLevel.INFO
    log_sensitive_data: bool = False
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 90
    
    # File Security
    upload_max_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        '.csv', '.json', '.parquet', '.xlsx'
    ])
    scan_uploads_for_malware: bool = True
    
    # Network Security
    force_https: bool = True
    hsts_enabled: bool = True
    csp_enabled: bool = True
    
    # Feature Flags
    advanced_security_features: bool = True
    vulnerability_scanning: bool = True
    threat_detection: bool = True
    
    def __post_init__(self):
        """Post-initialization security validation."""
        # Auto-generate JWT secret if not provided
        if not self.jwt_secret_key:
            self.jwt_secret_key = secrets.token_urlsafe(64)
        
        # Force production security settings
        if self.environment == Environment.PRODUCTION:
            self._enforce_production_security()
        
        # Validate configuration
        self._validate_configuration()
    
    def _enforce_production_security(self):
        """Enforce production security settings."""
        self.debug_enabled = False
        self.debug_logging = False
        self.log_level = LogLevel.INFO
        self.log_sensitive_data = False
        self.force_https = True
        self.hsts_enabled = True
        self.csp_enabled = True
        self.db_ssl_required = True
        self.db_encrypt_sensitive_data = True
        self.audit_log_enabled = True
        self.scan_uploads_for_malware = True
        self.advanced_security_features = True
        self.vulnerability_scanning = True
        self.threat_detection = True
        
        logger.info("Production security settings enforced")
    
    def _validate_configuration(self):
        """Validate security configuration."""
        errors = []
        
        # Validate JWT secret strength
        if len(self.jwt_secret_key) < 32:
            errors.append("JWT secret key must be at least 32 characters")
        
        # Validate password policy
        if self.password_min_length < 8:
            errors.append("Password minimum length must be at least 8 characters")
        
        # Validate rate limits
        if self.api_rate_limit_per_minute <= 0:
            errors.append("API rate limit must be positive")
        
        # Validate file size limits
        if self.max_request_size_mb > 1000:  # 1GB limit
            errors.append("Maximum request size too large (>1GB)")
        
        if errors:
            raise ValueError(f"Security configuration errors: {'; '.join(errors)}")
    
    def is_development(self) -> bool:
        """Check if in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.cors_allowed_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["*"],
        }
    
    def get_csp_header(self) -> str:
        """Get Content Security Policy header."""
        if not self.csp_enabled:
            return ""
        
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "environment": self.environment.value,
            "debug_enabled": self.debug_enabled,
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expiration_hours": self.jwt_expiration_hours,
            "password_min_length": self.password_min_length,
            "api_rate_limit_per_minute": self.api_rate_limit_per_minute,
            "max_request_size_mb": self.max_request_size_mb,
            "log_level": self.log_level.value,
            "force_https": self.force_https,
            "advanced_security_features": self.advanced_security_features,
        }

class SecurityConfigFactory:
    """Factory for creating security configurations."""
    
    @staticmethod
    def from_environment() -> SecurityConfig:
        """Create security config from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "production").lower()
        
        try:
            environment = Environment(env_name)
        except ValueError:
            logger.warning(f"Invalid environment '{env_name}', defaulting to production")
            environment = Environment.PRODUCTION
        
        config = SecurityConfig(
            environment=environment,
            
            # Debug settings (only allowed in development)
            debug_enabled=environment == Environment.DEVELOPMENT and 
                         os.getenv("DEBUG", "false").lower() == "true",
            debug_logging=environment == Environment.DEVELOPMENT and 
                         os.getenv("DEBUG_LOGGING", "false").lower() == "true",
            
            # Authentication
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            password_min_length=int(os.getenv("PASSWORD_MIN_LENGTH", "12")),
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
            
            # API Security
            api_rate_limit_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "100")),
            cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else [],
            max_request_size_mb=int(os.getenv("MAX_REQUEST_SIZE_MB", "10")),
            
            # Database Security
            db_ssl_required=os.getenv("DB_SSL_REQUIRED", "true").lower() == "true",
            db_encrypt_sensitive_data=os.getenv("DB_ENCRYPT_SENSITIVE_DATA", "true").lower() == "true",
            
            # Logging
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO").upper()),
            log_sensitive_data=environment == Environment.DEVELOPMENT and 
                              os.getenv("LOG_SENSITIVE_DATA", "false").lower() == "true",
            audit_log_enabled=os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true",
            
            # Network Security
            force_https=os.getenv("FORCE_HTTPS", "true").lower() == "true",
            hsts_enabled=os.getenv("HSTS_ENABLED", "true").lower() == "true",
            csp_enabled=os.getenv("CSP_ENABLED", "true").lower() == "true",
        )
        
        return config
    
    @staticmethod
    def for_testing() -> SecurityConfig:
        """Create security config for testing."""
        return SecurityConfig(
            environment=Environment.TESTING,
            debug_enabled=True,
            debug_logging=False,  # Keep false to avoid log noise
            jwt_secret_key="test_secret_key_" + secrets.token_urlsafe(32),
            db_ssl_required=False,
            force_https=False,
            audit_log_enabled=False,
        )
    
    @staticmethod
    def for_development() -> SecurityConfig:
        """Create security config for development."""
        return SecurityConfig(
            environment=Environment.DEVELOPMENT,
            debug_enabled=True,
            debug_logging=True,
            log_level=LogLevel.DEBUG,
            db_ssl_required=False,
            force_https=False,
            cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        )
    
    @staticmethod
    def for_production() -> SecurityConfig:
        """Create security config for production."""
        config = SecurityConfig(environment=Environment.PRODUCTION)
        # Production settings are enforced in __post_init__
        return config

# Singleton security config
_security_config: Optional[SecurityConfig] = None

def get_security_config() -> SecurityConfig:
    """Get global security configuration."""
    global _security_config
    
    if _security_config is None:
        _security_config = SecurityConfigFactory.from_environment()
    
    return _security_config

def set_security_config(config: SecurityConfig):
    """Set global security configuration."""
    global _security_config
    _security_config = config

# Security validation decorators
def require_production_environment(func):
    """Decorator to require production environment."""
    def wrapper(*args, **kwargs):
        config = get_security_config()
        if not config.is_production():
            raise RuntimeError("This operation requires production environment")
        return func(*args, **kwargs)
    return wrapper

def require_secure_configuration(func):
    """Decorator to require secure configuration."""
    def wrapper(*args, **kwargs):
        config = get_security_config()
        if config.debug_enabled and config.is_production():
            raise RuntimeError("Debug mode cannot be enabled in production")
        return func(*args, **kwargs)
    return wrapper

# Secure logging configuration
def configure_secure_logging():
    """Configure secure logging based on security settings."""
    config = get_security_config()
    
    # Set log level
    log_level = getattr(logging, config.log_level.value)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # In production, would add secure file handler with rotation
        ]
    )
    
    # Disable debug logging in production
    if config.is_production():
        logging.getLogger().setLevel(logging.INFO)
        
        # Disable verbose logging from third-party libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    
    logger.info(f"Secure logging configured for {config.environment.value} environment")

# Security header middleware
class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def get_headers(self) -> Dict[str, str]:
        """Get security headers."""
        headers = {}
        
        # HSTS header
        if self.config.hsts_enabled:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP header
        if self.config.csp_enabled:
            headers["Content-Security-Policy"] = self.config.get_csp_header()
        
        # Other security headers
        headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        })
        
        return headers

# Initialize secure logging on import
configure_secure_logging()