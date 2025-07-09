"""Advanced security infrastructure for Pynomaly.

This module provides comprehensive security features including:
- Input sanitization and validation
- SQL injection protection
- Data encryption
- Security headers enforcement
- Audit logging
- Security monitoring
"""

from .advanced_threat_detection import (
    AdvancedBehaviorAnalyzer,
    BehaviorProfile,
    DataExfiltrationDetector,
    ThreatIntelligence,
    ThreatIntelligenceDetector,
    create_advanced_threat_detectors,
)
from .audit_logger import (
    AuditEvent,
    AuditLogger,
    SecurityEvent,
    SecurityEventType,
    log_security_event,
)
from .encryption import (
    DataEncryption,
    EncryptionConfig,
    EncryptionService,
    FieldEncryption,
)
from .input_sanitizer import (
    InputSanitizer,
    SanitizationConfig,
    ValidationError,
    sanitize_input,
    validate_sql_safe,
)
from .middleware_integration import (
    SecurityMiddlewareStack,
    add_security_endpoints,
    decrypt_sensitive_data,
    encrypt_sensitive_data,
    sanitize_request_data,
    setup_security_middleware,
    validate_sql_query,
)
from .rate_limiting import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimiter,
    RateLimitManager,
    RateLimitScope,
    RateLimitStatus,
    RateLimitViolation,
    check_rate_limit,
    close_rate_limit_manager,
    get_rate_limit_manager,
    get_rate_limit_stats,
    rate_limit_context,
)
from .rate_limiting_decorators import (
    RateLimitDecoratorConfig,
    api_rate_limited,
    check_rate_limit_status,
    create_rate_limit_middleware,
    endpoint_rate_limited,
    rate_limited,
    user_rate_limited,
)
from .rate_limiting_decorators import rate_limit_context as rate_limit_decorator_context
from .security_headers import CSPConfig, SecurityHeaders, SecurityHeadersMiddleware
from .security_monitor import (
    SecurityAlert,
    SecurityMetrics,
    SecurityMonitor,
    ThreatDetector,
)
from .sql_protection import (
    QuerySanitizer,
    SafeQueryBuilder,
    SQLInjectionError,
    SQLInjectionProtector,
)
from .user_tracking import (
    RequestInfo,
    ResponseInfo,
    SensitiveDataFilter,
    UserAction,
    UserActionTracker,
    UserTrackingMiddleware,
)

__all__ = [
    # Input Sanitization
    "InputSanitizer",
    "SanitizationConfig",
    "ValidationError",
    "sanitize_input",
    "validate_sql_safe",
    # SQL Protection
    "SQLInjectionProtector",
    "QuerySanitizer",
    "SafeQueryBuilder",
    "SQLInjectionError",
    # Encryption
    "EncryptionService",
    "DataEncryption",
    "FieldEncryption",
    "EncryptionConfig",
    # Security Headers
    "SecurityHeadersMiddleware",
    "SecurityHeaders",
    "CSPConfig",
    # Audit Logging
    "AuditLogger",
    "SecurityEvent",
    "AuditEvent",
    "SecurityEventType",
    "log_security_event",
    # Security Monitoring
    "SecurityMonitor",
    "ThreatDetector",
    "SecurityAlert",
    "SecurityMetrics",
    # User Tracking
    "UserActionTracker",
    "UserTrackingMiddleware",
    "SensitiveDataFilter",
    "RequestInfo",
    "ResponseInfo",
    "UserAction",
    # Middleware Integration
    "SecurityMiddlewareStack",
    "setup_security_middleware",
    "add_security_endpoints",
    "sanitize_request_data",
    "validate_sql_query",
    "encrypt_sensitive_data",
    "decrypt_sensitive_data",
    # Advanced Threat Detection
    "AdvancedBehaviorAnalyzer",
    "ThreatIntelligenceDetector",
    "DataExfiltrationDetector",
    "ThreatIntelligence",
    "BehaviorProfile",
    "create_advanced_threat_detectors",
    # Rate Limiting
    "RateLimitAlgorithm",
    "RateLimitConfig",
    "RateLimitScope",
    "RateLimitStatus",
    "RateLimitViolation",
    "RateLimiter",
    "RateLimitManager",
    "check_rate_limit",
    "get_rate_limit_manager",
    "close_rate_limit_manager",
    "get_rate_limit_stats",
    "rate_limit_context",
    "RateLimitDecoratorConfig",
    "rate_limited",
    "user_rate_limited",
    "api_rate_limited",
    "endpoint_rate_limited",
    "rate_limit_decorator_context",
    "check_rate_limit_status",
    "create_rate_limit_middleware",
]
