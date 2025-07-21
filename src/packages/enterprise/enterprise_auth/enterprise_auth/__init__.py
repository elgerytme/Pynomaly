"""
Enterprise Authentication & Authorization Package for anomaly_detection

This package provides enterprise-grade authentication and authorization capabilities
including SSO, SAML, OAuth2, RBAC, and multi-tenancy support.
"""

from .domain.entities.user import User, UserRole, UserStatus
from .domain.entities.tenant import Tenant, TenantStatus, TenantPlan
from .domain.entities.permission import Permission, Role, RolePermission
from .application.services.auth_service import AuthService
from .application.services.tenant_service import TenantService
from .application.services.rbac_service import RBACService

__version__ = "0.4.0"
__author__ = "anomaly_detection Enterprise Team"
__email__ = "enterprise@anomaly_detection.org"

__all__ = [
    # Domain Entities
    "User",
    "UserRole", 
    "UserStatus",
    "Tenant",
    "TenantStatus",
    "TenantPlan",
    "Permission",
    "Role",
    "RolePermission",
    
    # Application Services
    "AuthService",
    "TenantService", 
    "RBACService",
]