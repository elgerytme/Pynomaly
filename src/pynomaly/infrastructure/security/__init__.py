"""Advanced security infrastructure for Pynomaly.

This module provides comprehensive security features including:
- Input sanitization and validation
- SQL injection protection
- Data encryption
- Security headers enforcement
- Audit logging
- Security monitoring
"""

from .input_sanitizer import (
    InputSanitizer,
    SanitizationConfig,
    ValidationError,
    sanitize_input,
    validate_sql_safe,
)

from .sql_protection import (
    SQLInjectionProtector,
    QuerySanitizer,
    SafeQueryBuilder,
    SQLInjectionError,
)

from .encryption import (
    EncryptionService,
    DataEncryption,
    FieldEncryption,
    EncryptionConfig,
)

from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeaders,
    CSPConfig,
)

from .audit_logger import (
    AuditLogger,
    SecurityEvent,
    AuditEvent,
    SecurityEventType,
    log_security_event,
)

from .security_monitor import (
    SecurityMonitor,
    ThreatDetector,
    SecurityAlert,
    SecurityMetrics,
)

from .user_tracking import (
    UserActionTracker,
    UserTrackingMiddleware,
    SensitiveDataFilter,
    RequestInfo,
    ResponseInfo,
    UserAction,
)

from .middleware_integration import (
    SecurityMiddlewareStack,
    setup_security_middleware,
    add_security_endpoints,
    sanitize_request_data,
    validate_sql_query,
    encrypt_sensitive_data,
    decrypt_sensitive_data,
)

from .advanced_threat_detection import (
    AdvancedBehaviorAnalyzer,
    ThreatIntelligenceDetector,
    DataExfiltrationDetector,
    ThreatIntelligence,
    BehaviorProfile,
    create_advanced_threat_detectors,
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
]