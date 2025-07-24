"""
Enterprise Governance Package

Comprehensive governance, audit, compliance, and SLA management
for enterprise environments with regulatory compliance support.
"""

# Domain entities
from .domain.entities.audit_log import (
    AuditLog, AuditQuery, AuditStatistics, AuditRetentionPolicy,
    AuditEventType, AuditSeverity, AuditStatus
)

from .domain.entities.compliance import (
    ComplianceControl, ComplianceAssessment, ComplianceReport, DataPrivacyRecord,
    ComplianceFramework, ComplianceStatus, ControlStatus, EvidenceType
)

from .domain.entities.sla import (
    ServiceLevelAgreement, SLAMetric, SLAViolation,
    SLAStatus, SLAType, SLAMetricType, SLAViolationSeverity
)

# Application services
from .application.services.governance_service import GovernanceService

# DTOs
from .application.dto.governance_dto import (
    # Audit DTOs
    AuditLogRequest, AuditLogResponse,
    AuditSearchRequest, AuditSearchResponse,
    AuditReportRequest,
    
    # Compliance DTOs
    ComplianceAssessmentRequest, ComplianceAssessmentResponse,
    ControlUpdateRequest, ControlResponse,
    ComplianceReportRequest,
    
    # SLA DTOs
    SLARequest, SLAResponse,
    SLAMetricRequest, SLAMetricResponse,
    MetricMeasurementRequest,
    SLAViolationResponse,
    SLAComplianceRequest, SLAComplianceResponse,
    
    # Data Privacy DTOs
    DataPrivacyRequest, DataRightsRequest, DataPrivacyResponse,
    
    # General DTOs
    GovernanceStatsResponse
)

__version__ = "0.1.0"
__author__ = "anomaly_detection Enterprise Team"
__email__ = "enterprise@anomaly_detection.org"

__all__ = [
    # Domain entities
    "AuditLog", "AuditQuery", "AuditStatistics", "AuditRetentionPolicy",
    "AuditEventType", "AuditSeverity", "AuditStatus",
    
    "ComplianceControl", "ComplianceAssessment", "ComplianceReport", "DataPrivacyRecord",
    "ComplianceFramework", "ComplianceStatus", "ControlStatus", "EvidenceType",
    
    "ServiceLevelAgreement", "SLAMetric", "SLAViolation",
    "SLAStatus", "SLAType", "SLAMetricType", "SLAViolationSeverity",
    
    # Application services
    "GovernanceService",
    
    # DTOs
    "AuditLogRequest", "AuditLogResponse",
    "AuditSearchRequest", "AuditSearchResponse",
    "AuditReportRequest",
    
    "ComplianceAssessmentRequest", "ComplianceAssessmentResponse",
    "ControlUpdateRequest", "ControlResponse",
    "ComplianceReportRequest",
    
    "SLARequest", "SLAResponse",
    "SLAMetricRequest", "SLAMetricResponse",
    "MetricMeasurementRequest",
    "SLAViolationResponse",
    "SLAComplianceRequest", "SLAComplianceResponse",
    
    "DataPrivacyRequest", "DataRightsRequest", "DataPrivacyResponse",
    "GovernanceStatsResponse",
]

# Package metadata
PACKAGE_INFO = {
    "name": "anomaly_detection-enterprise-governance",
    "version": __version__,
    "description": "Enterprise governance, audit, and compliance for anomaly_detection",
    "author": __author__,
    "email": __email__,
    "features": [
        "Comprehensive audit logging and trail management",
        "Multi-framework compliance assessment (SOC2, GDPR, ISO27001, HIPAA, PCI-DSS)",
        "Service Level Agreement monitoring and violation management",
        "Data privacy and protection compliance (GDPR, CCPA)",
        "Automated compliance reporting and documentation",
        "Real-time governance dashboards and metrics",
        "Integration with enterprise security and monitoring systems"
    ],
    "compliance_frameworks": [
        "SOC 2 Type II",
        "GDPR (General Data Protection Regulation)",
        "ISO 27001",
        "HIPAA",
        "PCI-DSS",
        "CCPA (California Consumer Privacy Act)",
        "FedRAMP",
        "NIST Cybersecurity Framework"
    ],
    "integrations": [
        "Enterprise SIEM systems",
        "Compliance management monorepos", 
        "Service monitoring tools",
        "Notification systems (Slack, Teams, PagerDuty)",
        "Report generation and document management",
        "Identity and access management systems"
    ]
}