"""
Security hardening module for Software API.

This module provides advanced security measures including:
- Input validation and sanitization
- Authentication and authorization
- Rate limiting and DDoS protection
- Security headers and CORS policies
- SQL injection and XSS prevention
- Encryption at rest and in transit
- Security monitoring and logging
"""

from .authentication import AuthenticationManager, JWTManager
from .authorization import AuthorizationManager, RoleBasedAccessControl
from .encryption import EncryptionManager
from .input_validation import InputValidator, SecuritySanitizer
from .rate_limiting import DDoSProtection, RateLimiter
from .security_headers import SecurityHeaders
from .security_monitoring import SecurityMonitor
from .vulnerability_scanner import VulnerabilityScanner

__all__ = [
    "AuthenticationManager",
    "JWTManager",
    "AuthorizationManager",
    "RoleBasedAccessControl",
    "InputValidator",
    "SecuritySanitizer",
    "RateLimiter",
    "DDoSProtection",
    "SecurityHeaders",
    "EncryptionManager",
    "SecurityMonitor",
    "VulnerabilityScanner",
]
