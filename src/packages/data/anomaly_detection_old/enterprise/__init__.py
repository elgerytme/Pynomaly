"""
Enterprise Features for Pynomaly Detection
==========================================

This module provides enterprise-grade features including:
- Multi-tenancy and tenant isolation
- Role-based access control (RBAC)
- Audit logging and compliance tracking
- Enterprise security features
- API rate limiting and throttling
- Enterprise dashboard and reporting
"""

from .multi_tenancy import TenantManager, TenantIsolationService
from .rbac import RoleBasedAccessControl, PermissionManager
from .audit import AuditLogger, ComplianceTracker
from .security import SecurityManager, EncryptionService
from .rate_limiting import RateLimiter, ThrottlingService
from .dashboard import EnterpriseDashboard, ReportingService

__all__ = [
    # Multi-tenancy
    'TenantManager',
    'TenantIsolationService',
    
    # RBAC
    'RoleBasedAccessControl',
    'PermissionManager',
    
    # Audit and Compliance
    'AuditLogger',
    'ComplianceTracker',
    
    # Security
    'SecurityManager',
    'EncryptionService',
    
    # Rate Limiting
    'RateLimiter',
    'ThrottlingService',
    
    # Dashboard and Reporting
    'EnterpriseDashboard',
    'ReportingService'
]

# Enterprise Features Version
__version__ = "1.0.0"

def get_enterprise_features_info():
    """Get enterprise features information."""
    return {
        "version": __version__,
        "capabilities": {
            "multi_tenancy": "Complete tenant isolation with resource quotas",
            "rbac": "Role-based access control with fine-grained permissions",
            "audit": "Comprehensive audit logging and compliance tracking",
            "security": "Enterprise-grade security with encryption",
            "rate_limiting": "API rate limiting and throttling",
            "dashboard": "Enterprise dashboard with advanced reporting"
        },
        "compliance_standards": [
            "SOC 2", "GDPR", "HIPAA", "PCI DSS", "ISO 27001"
        ]
    }