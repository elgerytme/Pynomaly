"""Data Quality Package with Enterprise Security and Compliance.

This package provides comprehensive data quality capabilities with enterprise-grade
security and compliance features including:

- PII detection and automatic masking
- Privacy-preserving analytics with differential privacy
- Consent management and GDPR compliance
- Privacy impact assessments
- Multi-framework compliance (GDPR, HIPAA, SOX, CCPA, PCI-DSS)
- Role-based and attribute-based access control
- Multi-factor authentication and SSO integration
- Comprehensive audit trails and monitoring
- Threat detection and incident response
- End-to-end encryption and key management
- Secure multi-tenancy
"""

from .application.services.pii_detection_service import (
    PIIDetectionService,
    PIIAuditService
)
from .application.services.privacy_analytics_service import (
    DifferentialPrivacyService,
    PrivacyPreservingAnalyticsService,
    SecureAggregationService,
    HomomorphicEncryptionService
)
from .application.services.consent_management_service import (
    ConsentManagementService,
    ConsentOrchestrationService
)
from .application.services.privacy_impact_assessment_service import (
    PrivacyImpactAssessmentService
)
from .application.services.compliance_framework_service import (
    ComplianceFrameworkService,
    ComplianceOrchestrationService
)
from .application.services.access_control_service import (
    RoleBasedAccessControlService,
    AttributeBasedAccessControlService,
    AccessControlOrchestrationService
)
from .application.services.security_orchestration_service import (
    SecurityOrchestrationService,
    EncryptionService,
    AuthenticationService,
    SessionManagementService,
    ThreatDetectionService,
    AuditService,
    IncidentResponseService
)

from .domain.entities.security_entity import (
    PIIType,
    PrivacyLevel,
    ComplianceFramework,
    SecurityEventType,
    PIIDetectionResult,
    ConsentRecord,
    PrivacyImpactAssessment,
    SecurityEvent,
    AccessPermission,
    UserRole,
    EncryptionConfig,
    AuditRecord,
    ComplianceAssessment,
    ThreatDetectionResult
)

# Version information
__version__ = "1.0.0"
__author__ = "Pynomaly Security Team"
__description__ = "Enterprise Data Quality with Security and Compliance"

# Export all security services
__all__ = [
    # Services
    "PIIDetectionService",
    "PIIAuditService",
    "DifferentialPrivacyService",
    "PrivacyPreservingAnalyticsService",
    "SecureAggregationService",
    "HomomorphicEncryptionService",
    "ConsentManagementService",
    "ConsentOrchestrationService",
    "PrivacyImpactAssessmentService",
    "ComplianceFrameworkService",
    "ComplianceOrchestrationService",
    "RoleBasedAccessControlService",
    "AttributeBasedAccessControlService",
    "AccessControlOrchestrationService",
    "SecurityOrchestrationService",
    "EncryptionService",
    "AuthenticationService",
    "SessionManagementService",
    "ThreatDetectionService",
    "AuditService",
    "IncidentResponseService",
    
    # Entities
    "PIIType",
    "PrivacyLevel",
    "ComplianceFramework",
    "SecurityEventType",
    "PIIDetectionResult",
    "ConsentRecord",
    "PrivacyImpactAssessment",
    "SecurityEvent",
    "AccessPermission",
    "UserRole",
    "EncryptionConfig",
    "AuditRecord",
    "ComplianceAssessment",
    "ThreatDetectionResult",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]

# Security configuration
SECURITY_CONFIG = {
    "encryption": {
        "default_algorithm": "AES-256-GCM",
        "key_rotation_days": 90,
        "at_rest_encryption": True,
        "in_transit_encryption": True
    },
    "authentication": {
        "mfa_required": True,
        "session_timeout_minutes": 30,
        "password_policy": {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": True
        }
    },
    "authorization": {
        "rbac_enabled": True,
        "abac_enabled": True,
        "default_deny": True
    },
    "compliance": {
        "frameworks": ["GDPR", "HIPAA", "SOX", "CCPA", "PCI-DSS"],
        "audit_retention_days": 2555,  # 7 years
        "real_time_monitoring": True
    },
    "privacy": {
        "differential_privacy": True,
        "pii_detection": True,
        "consent_management": True,
        "privacy_impact_assessment": True
    }
}

# Initialize security orchestrator
def create_security_orchestrator():
    """Create and configure security orchestrator."""
    return SecurityOrchestrationService()

# Initialize compliance framework
def create_compliance_framework():
    """Create and configure compliance framework."""
    return ComplianceFrameworkService()

# Initialize access control
def create_access_control():
    """Create and configure access control services."""
    rbac = RoleBasedAccessControlService()
    abac = AttributeBasedAccessControlService()
    return AccessControlOrchestrationService(rbac, abac)

# Initialize PII detection
def create_pii_detection():
    """Create and configure PII detection service."""
    return PIIDetectionService()

# Initialize privacy analytics
def create_privacy_analytics(privacy_level=PrivacyLevel.CONFIDENTIAL):
    """Create and configure privacy analytics service."""
    return PrivacyPreservingAnalyticsService(privacy_level)

# Initialize consent management
def create_consent_management():
    """Create and configure consent management service."""
    return ConsentManagementService()

# Quick setup function
def setup_enterprise_security():
    """Set up complete enterprise security suite."""
    orchestrator = create_security_orchestrator()
    
    # Configure default security policies
    orchestrator.abac_service.create_policy(
        policy_id="default_deny",
        policy_config={
            "type": "access_control",
            "effect": "deny",
            "target": {"action": ["*"]},
            "condition": {
                "type": "attribute",
                "attribute": "subject.authenticated",
                "operator": "not_equals",
                "value": True
            }
        }
    )
    
    orchestrator.abac_service.create_policy(
        policy_id="admin_access",
        policy_config={
            "type": "access_control",
            "effect": "permit",
            "target": {"subject": {"roles": ["admin"]}},
            "condition": {
                "type": "and",
                "conditions": [
                    {
                        "type": "attribute",
                        "attribute": "subject.authenticated",
                        "operator": "equals",
                        "value": True
                    },
                    {
                        "type": "time",
                        "time_of_day": {"start_hour": 6, "end_hour": 22}
                    }
                ]
            }
        }
    )
    
    return orchestrator