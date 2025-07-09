#!/usr/bin/env python3
"""
Enterprise Package for Pynomaly.
This package provides enterprise-grade features including multi-tenancy, audit logging, and compliance.
"""

from .multi_tenancy import (
    MultiTenantManager, TenantInfo, TenantUserInfo, TenantStatus, UserRole, ResourceType,
    TenantCreateRequest, TenantUpdateRequest, UserCreateRequest, LoginRequest, TokenResponse,
    get_multi_tenant_manager, get_current_user, get_current_tenant, require_permission
)

from .audit_logging import (
    AuditLogger, AuditEventInfo, AuditEventCreate, AuditQuery,
    AuditAction, AuditStatus, ComplianceLevel, SensitivityLevel,
    audit_log, get_audit_logger
)

from .enterprise_service import (
    EnterpriseService, EnterpriseHealthResponse, EnterpriseDashboardResponse,
    ComplianceReportRequest, ComplianceReportResponse, enterprise_service, router
)

__all__ = [
    # Multi-tenancy
    "MultiTenantManager", "TenantInfo", "TenantUserInfo", "TenantStatus", "UserRole", "ResourceType",
    "TenantCreateRequest", "TenantUpdateRequest", "UserCreateRequest", "LoginRequest", "TokenResponse",
    "get_multi_tenant_manager", "get_current_user", "get_current_tenant", "require_permission",
    
    # Audit logging
    "AuditLogger", "AuditEventInfo", "AuditEventCreate", "AuditQuery",
    "AuditAction", "AuditStatus", "ComplianceLevel", "SensitivityLevel",
    "audit_log", "get_audit_logger",
    
    # Enterprise service
    "EnterpriseService", "EnterpriseHealthResponse", "EnterpriseDashboardResponse",
    "ComplianceReportRequest", "ComplianceReportResponse", "enterprise_service", "router"
]

# Version information
__version__ = "1.0.0"
__author__ = "Pynomaly Team"
__description__ = "Enterprise-grade multi-tenancy and audit logging for Pynomaly"