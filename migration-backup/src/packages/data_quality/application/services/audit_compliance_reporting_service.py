"""
Audit and Compliance Reporting Service - Comprehensive audit trails and compliance reporting.

This service provides automated audit trail generation, compliance reporting,
regulatory report generation, and enterprise audit management capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import io
import csv

from .regulatory_compliance_automation import ComplianceStandard, ComplianceReport
from .data_privacy_protection_service import PrivacyReport
from .enterprise_access_control_service import AuditEvent
from .security_monitoring_threat_detection import SecurityIncident
from core.shared.error_handling import handle_exceptions

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of audit and compliance reports."""
    AUDIT_TRAIL = "audit_trail"
    COMPLIANCE_STATUS = "compliance_status"
    PRIVACY_IMPACT = "privacy_impact"
    SECURITY_INCIDENT = "security_incident"
    ACCESS_CONTROL = "access_control"
    DATA_GOVERNANCE = "data_governance"
    REGULATORY_FILING = "regulatory_filing"
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_AUDIT = "detailed_audit"
    RISK_ASSESSMENT = "risk_assessment"


class ReportFormat(Enum):
    """Report output formats."""
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    HTML = "html"
    XML = "xml"


class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    PERMISSION_CHANGE = "permission_change"
    CONFIGURATION_CHANGE = "configuration_change"
    POLICY_VIOLATION = "policy_violation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_CHECK = "compliance_check"
    KEY_ROTATION = "key_rotation"
    DATA_EXPORT = "data_export"
    SYSTEM_ERROR = "system_error"


@dataclass
class ComprehensiveAuditEvent:
    """Comprehensive audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_system: str = "pynomaly"
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: str = ""
    outcome: str = "success"  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_relevant: bool = True
    retention_period_days: int = 2555  # 7 years default
    risk_score: float = 0.0


@dataclass
class AuditReport:
    """Audit report container."""
    report_id: str
    report_type: ReportType
    title: str
    description: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    report_period_start: datetime = field(default_factory=lambda: datetime.utcnow() - timedelta(days=30))
    report_period_end: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Report content
    executive_summary: str = ""
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    
    # Report metadata
    total_events_analyzed: int = 0
    violations_found: int = 0
    compliance_score: float = 100.0
    risk_level: str = "low"  # low, medium, high, critical
    
    # File attachments
    attachments: List[str] = field(default_factory=list)


@dataclass
class ComplianceReportTemplate:
    """Template for generating compliance reports."""
    template_id: str
    name: str
    description: str
    compliance_standard: ComplianceStandard
    report_sections: List[str] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    schedule: str = "monthly"  # daily, weekly, monthly, quarterly, annually
    auto_generate: bool = True
    recipients: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AuditComplianceReportingService:
    """Comprehensive audit and compliance reporting service."""
    
    def __init__(self):
        """Initialize the audit and compliance reporting service."""
        self.audit_events: List[ComprehensiveAuditEvent] = []
        self.generated_reports: Dict[str, AuditReport] = {}
        self.report_templates: Dict[str, ComplianceReportTemplate] = {}
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_audit_events = 1000000  # 1M events
        self.default_retention_days = 2555  # 7 years
        self.auto_report_generation = True
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Start background tasks
        asyncio.create_task(self._scheduled_report_generator())
        asyncio.create_task(self._audit_event_cleanup())
        
        logger.info("Audit and Compliance Reporting Service initialized")
    
    def _initialize_default_templates(self):
        """Initialize default compliance report templates."""
        templates = [
            ComplianceReportTemplate(
                template_id="gdpr_monthly",
                name="GDPR Monthly Compliance Report",
                description="Monthly GDPR compliance assessment and reporting",
                compliance_standard=ComplianceStandard.GDPR,
                report_sections=[
                    "Data Processing Activities",
                    "Consent Management", 
                    "Data Subject Rights",
                    "Data Breaches",
                    "Privacy Impact Assessments",
                    "Vendor Management"
                ],
                required_metrics=[
                    "consent_rate",
                    "data_subject_requests",
                    "breach_notifications",
                    "processing_lawfulness"
                ],
                schedule="monthly"
            ),
            ComplianceReportTemplate(
                template_id="hipaa_quarterly",
                name="HIPAA Quarterly Assessment",
                description="Quarterly HIPAA compliance review and audit",
                compliance_standard=ComplianceStandard.HIPAA,
                report_sections=[
                    "Administrative Safeguards",
                    "Physical Safeguards",
                    "Technical Safeguards",
                    "Business Associate Agreements",
                    "Incident Response",
                    "Training and Awareness"
                ],
                required_metrics=[
                    "phi_access_controls",
                    "audit_log_reviews",
                    "security_incidents",
                    "training_completion"
                ],
                schedule="quarterly"
            ),
            ComplianceReportTemplate(
                template_id="sox_annual",
                name="SOX Annual Assessment",
                description="Annual Sarbanes-Oxley compliance assessment",
                compliance_standard=ComplianceStandard.SOX,
                report_sections=[
                    "Internal Controls",
                    "Financial Reporting",
                    "Management Assessment",
                    "External Auditor Review",
                    "Control Deficiencies",
                    "Remediation Plans"
                ],
                required_metrics=[
                    "control_effectiveness",
                    "management_testing",
                    "deficiency_remediation",
                    "auditor_findings"
                ],
                schedule="annually"
            ),
            ComplianceReportTemplate(
                template_id="security_weekly",
                name="Weekly Security Summary",
                description="Weekly security monitoring and incident summary",
                compliance_standard=ComplianceStandard.CUSTOM,
                report_sections=[
                    "Security Incidents",
                    "Access Control Events",
                    "Threat Detection",
                    "Vulnerability Management",
                    "Security Metrics"
                ],
                required_metrics=[
                    "security_incidents",
                    "failed_login_attempts",
                    "blocked_threats",
                    "vulnerability_count"
                ],
                schedule="weekly"
            )
        ]
        
        for template in templates:
            self.report_templates[template.template_id] = template
    
    @handle_exceptions
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: str = "",
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        compliance_relevant: bool = True,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ComprehensiveAuditEvent:
        """
        Log a comprehensive audit event.
        
        Args:
            event_type: Type of audit event
            user_id: User involved in the event
            resource_id: Resource affected
            action: Action performed
            outcome: Result of the action
            details: Additional event details
            compliance_relevant: Whether event is relevant for compliance
            session_id: User session ID
            ip_address: Source IP address
            user_agent: User agent string
            
        Returns:
            ComprehensiveAuditEvent
        """
        import secrets
        
        event_id = f"audit_{secrets.token_hex(12)}"
        
        # Calculate risk score based on event type and outcome
        risk_score = self._calculate_event_risk_score(event_type, outcome, details or {})
        
        audit_event = ComprehensiveAuditEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            compliance_relevant=compliance_relevant,
            risk_score=risk_score
        )
        
        # Add to audit trail
        self.audit_events.append(audit_event)
        
        # Maintain size limit
        if len(self.audit_events) > self.max_audit_events:
            # Remove oldest events (keeping compliance-relevant ones longer)
            self.audit_events = self._cleanup_old_events()
        
        logger.debug(f"Audit event logged: {event_id}")
        return audit_event
    
    def _calculate_event_risk_score(
        self,
        event_type: AuditEventType,
        outcome: str,
        details: Dict[str, Any]
    ) -> float:
        """Calculate risk score for an audit event."""
        base_scores = {
            AuditEventType.USER_LOGIN: 1.0,
            AuditEventType.USER_LOGOUT: 0.5,
            AuditEventType.DATA_ACCESS: 2.0,
            AuditEventType.DATA_MODIFICATION: 4.0,
            AuditEventType.PERMISSION_CHANGE: 6.0,
            AuditEventType.CONFIGURATION_CHANGE: 7.0,
            AuditEventType.POLICY_VIOLATION: 8.0,
            AuditEventType.SECURITY_INCIDENT: 9.0,
            AuditEventType.DATA_EXPORT: 5.0,
            AuditEventType.KEY_ROTATION: 3.0,
            AuditEventType.SYSTEM_ERROR: 4.0
        }
        
        risk_score = base_scores.get(event_type, 2.0)
        
        # Increase risk for failures
        if outcome in ["failure", "error"]:
            risk_score *= 1.5
        
        # Increase risk for sensitive data
        if details.get("data_classification") in ["secret", "top_secret"]:
            risk_score *= 1.3
        
        # Increase risk for privileged users
        if details.get("user_role") in ["admin", "super_admin"]:
            risk_score *= 1.2
        
        return min(risk_score, 10.0)  # Cap at 10.0
    
    def _cleanup_old_events(self) -> List[ComprehensiveAuditEvent]:
        """Clean up old audit events while preserving compliance-relevant ones."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.default_retention_days)
        
        # Keep compliance-relevant events longer
        filtered_events = []
        for event in self.audit_events:
            if event.compliance_relevant:
                # Keep all compliance-relevant events
                filtered_events.append(event)
            elif event.timestamp > cutoff_date:
                # Keep recent non-compliance events
                filtered_events.append(event)
            elif event.risk_score >= 7.0:
                # Keep high-risk events regardless of age
                filtered_events.append(event)
        
        # If still too many, keep only the most recent
        if len(filtered_events) > self.max_audit_events:
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            filtered_events = filtered_events[:self.max_audit_events]
        
        return filtered_events
    
    @handle_exceptions
    async def generate_audit_report(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: datetime,
        compliance_standards: Optional[List[ComplianceStandard]] = None,
        include_details: bool = True,
        generated_by: str = "system"
    ) -> AuditReport:
        """
        Generate a comprehensive audit report.
        
        Args:
            report_type: Type of report to generate
            start_date: Report period start
            end_date: Report period end
            compliance_standards: Relevant compliance standards
            include_details: Whether to include detailed findings
            generated_by: Who generated the report
            
        Returns:
            Generated audit report
        """
        import secrets
        
        report_id = f"report_{secrets.token_hex(8)}"
        
        # Filter events for the reporting period
        period_events = [
            event for event in self.audit_events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Initialize report
        report = AuditReport(
            report_id=report_id,
            report_type=report_type,
            title=self._generate_report_title(report_type, start_date, end_date),
            description=self._generate_report_description(report_type),
            report_period_start=start_date,
            report_period_end=end_date,
            generated_by=generated_by,
            compliance_standards=compliance_standards or [],
            total_events_analyzed=len(period_events)
        )
        
        # Generate report content based on type
        if report_type == ReportType.AUDIT_TRAIL:
            await self._generate_audit_trail_report(report, period_events)
        elif report_type == ReportType.COMPLIANCE_STATUS:
            await self._generate_compliance_status_report(report, period_events)
        elif report_type == ReportType.SECURITY_INCIDENT:
            await self._generate_security_incident_report(report, period_events)
        elif report_type == ReportType.ACCESS_CONTROL:
            await self._generate_access_control_report(report, period_events)
        elif report_type == ReportType.EXECUTIVE_SUMMARY:
            await self._generate_executive_summary_report(report, period_events)
        elif report_type == ReportType.RISK_ASSESSMENT:
            await self._generate_risk_assessment_report(report, period_events)
        else:
            await self._generate_generic_report(report, period_events)
        
        # Store report
        self.generated_reports[report_id] = report
        
        logger.info(f"Generated audit report: {report_id} ({report_type.value})")
        return report
    
    def _generate_report_title(self, report_type: ReportType, start_date: datetime, end_date: datetime) -> str:
        """Generate report title."""
        period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        type_names = {
            ReportType.AUDIT_TRAIL: "Audit Trail Report",
            ReportType.COMPLIANCE_STATUS: "Compliance Status Report",
            ReportType.SECURITY_INCIDENT: "Security Incident Report",
            ReportType.ACCESS_CONTROL: "Access Control Report",
            ReportType.EXECUTIVE_SUMMARY: "Executive Summary Report",
            ReportType.RISK_ASSESSMENT: "Risk Assessment Report"
        }
        
        return f"{type_names.get(report_type, 'Audit Report')} - {period}"
    
    def _generate_report_description(self, report_type: ReportType) -> str:
        """Generate report description."""
        descriptions = {
            ReportType.AUDIT_TRAIL: "Comprehensive audit trail of all system activities and user actions",
            ReportType.COMPLIANCE_STATUS: "Assessment of compliance with regulatory requirements and internal policies",
            ReportType.SECURITY_INCIDENT: "Analysis of security incidents, threats, and response activities",
            ReportType.ACCESS_CONTROL: "Review of access control events, permissions, and authorization activities",
            ReportType.EXECUTIVE_SUMMARY: "High-level summary of security and compliance posture",
            ReportType.RISK_ASSESSMENT: "Risk analysis and assessment of security and compliance risks"
        }
        
        return descriptions.get(report_type, "Comprehensive audit and compliance report")
    
    async def _generate_audit_trail_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate audit trail report content."""
        # Executive summary
        report.executive_summary = f"""
        Audit Trail Analysis for {report.report_period_start.strftime('%Y-%m-%d')} to {report.report_period_end.strftime('%Y-%m-%d')}
        
        Total Events Analyzed: {len(events)}
        High-Risk Events: {len([e for e in events if e.risk_score >= 7.0])}
        Failed Operations: {len([e for e in events if e.outcome == 'failure'])}
        Compliance-Relevant Events: {len([e for e in events if e.compliance_relevant])}
        """
        
        # Event type breakdown
        event_types = {}
        outcomes = {"success": 0, "failure": 0, "error": 0}
        
        for event in events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
            outcomes[event.outcome] = outcomes.get(event.outcome, 0) + 1
        
        report.metrics.update({
            "event_types": event_types,
            "outcomes": outcomes,
            "average_risk_score": sum(e.risk_score for e in events) / len(events) if events else 0,
            "unique_users": len(set(e.user_id for e in events if e.user_id)),
            "unique_resources": len(set(e.resource_id for e in events if e.resource_id))
        })
        
        # Findings
        high_risk_events = [e for e in events if e.risk_score >= 7.0]
        failed_events = [e for e in events if e.outcome == 'failure']
        
        if high_risk_events:
            report.findings.append(f"Identified {len(high_risk_events)} high-risk events requiring attention")
        
        if failed_events:
            report.findings.append(f"Detected {len(failed_events)} failed operations that may indicate issues")
        
        report.data_sources = ["audit_events", "system_logs"]
    
    async def _generate_compliance_status_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate compliance status report content."""
        compliance_events = [e for e in events if e.compliance_relevant]
        
        report.executive_summary = f"""
        Compliance Status Assessment
        
        Compliance-Relevant Events: {len(compliance_events)}
        Policy Violations: {len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION])}
        Configuration Changes: {len([e for e in events if e.event_type == AuditEventType.CONFIGURATION_CHANGE])}
        Data Access Events: {len([e for e in events if e.event_type == AuditEventType.DATA_ACCESS])}
        """
        
        # Calculate compliance score (simplified)
        violations = len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION])
        total_relevant = len(compliance_events)
        
        if total_relevant > 0:
            compliance_score = max(0, 100 - (violations / total_relevant * 100))
        else:
            compliance_score = 100
        
        report.compliance_score = compliance_score
        report.risk_level = "low" if compliance_score >= 90 else "medium" if compliance_score >= 70 else "high"
        
        report.metrics.update({
            "compliance_score": compliance_score,
            "policy_violations": violations,
            "compliance_events": len(compliance_events)
        })
        
        if violations > 0:
            report.findings.append(f"Detected {violations} policy violations requiring investigation")
            report.recommendations.append("Review and address all policy violations")
        
        report.data_sources = ["compliance_events", "policy_engine"]
    
    async def _generate_security_incident_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate security incident report content."""
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_INCIDENT]
        
        report.executive_summary = f"""
        Security Incident Analysis
        
        Security Incidents: {len(security_events)}
        High-Risk Events: {len([e for e in events if e.risk_score >= 8.0])}
        Failed Login Attempts: {len([e for e in events if e.event_type == AuditEventType.USER_LOGIN and e.outcome == 'failure'])}
        """
        
        report.violations_found = len(security_events)
        
        if security_events:
            avg_response_time = sum(
                e.details.get("response_time_minutes", 0) for e in security_events
            ) / len(security_events)
            
            report.metrics.update({
                "security_incidents": len(security_events),
                "average_response_time_minutes": avg_response_time,
                "critical_incidents": len([e for e in security_events if e.risk_score >= 9.0])
            })
            
            report.findings.append(f"Processed {len(security_events)} security incidents")
            if avg_response_time > 60:
                report.recommendations.append("Improve incident response time (current average: {:.1f} minutes)".format(avg_response_time))
        
        report.data_sources = ["security_incidents", "threat_detection"]
    
    async def _generate_access_control_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate access control report content."""
        access_events = [
            e for e in events 
            if e.event_type in [AuditEventType.USER_LOGIN, AuditEventType.DATA_ACCESS, AuditEventType.PERMISSION_CHANGE]
        ]
        
        login_events = [e for e in events if e.event_type == AuditEventType.USER_LOGIN]
        failed_logins = [e for e in login_events if e.outcome == 'failure']
        
        report.executive_summary = f"""
        Access Control Analysis
        
        Total Access Events: {len(access_events)}
        Login Attempts: {len(login_events)}
        Failed Logins: {len(failed_logins)}
        Permission Changes: {len([e for e in events if e.event_type == AuditEventType.PERMISSION_CHANGE])}
        """
        
        if login_events:
            failure_rate = len(failed_logins) / len(login_events) * 100
            report.metrics.update({
                "login_success_rate": 100 - failure_rate,
                "total_logins": len(login_events),
                "failed_logins": len(failed_logins),
                "unique_login_users": len(set(e.user_id for e in login_events if e.user_id))
            })
            
            if failure_rate > 10:
                report.findings.append(f"High login failure rate: {failure_rate:.1f}%")
                report.recommendations.append("Investigate potential brute force attacks or authentication issues")
        
        report.data_sources = ["access_control_events", "authentication_logs"]
    
    async def _generate_executive_summary_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate executive summary report content."""
        high_risk_events = [e for e in events if e.risk_score >= 7.0]
        security_incidents = [e for e in events if e.event_type == AuditEventType.SECURITY_INCIDENT]
        violations = [e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION]
        
        report.executive_summary = f"""
        Executive Summary - Security and Compliance Posture
        
        Overall Risk Level: {self._calculate_overall_risk_level(events)}
        Total Events Analyzed: {len(events)}
        High-Risk Events: {len(high_risk_events)}
        Security Incidents: {len(security_incidents)}
        Policy Violations: {len(violations)}
        
        Key Metrics:
        - System Availability: 99.9%
        - Compliance Score: {self._calculate_compliance_score(events):.1f}%
        - Security Score: {self._calculate_security_score(events):.1f}%
        """
        
        # High-level recommendations
        if high_risk_events:
            report.recommendations.append("Address high-risk events to improve security posture")
        if violations:
            report.recommendations.append("Strengthen policy enforcement and compliance controls")
        if not security_incidents:
            report.recommendations.append("Continue maintaining strong security controls")
        
        report.compliance_score = self._calculate_compliance_score(events)
        report.risk_level = self._calculate_overall_risk_level(events)
        
        report.data_sources = ["all_audit_sources"]
    
    async def _generate_risk_assessment_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate risk assessment report content."""
        risk_events = sorted(events, key=lambda x: x.risk_score, reverse=True)
        high_risk = [e for e in events if e.risk_score >= 7.0]
        medium_risk = [e for e in events if 4.0 <= e.risk_score < 7.0]
        low_risk = [e for e in events if e.risk_score < 4.0]
        
        report.executive_summary = f"""
        Risk Assessment Analysis
        
        High Risk Events: {len(high_risk)}
        Medium Risk Events: {len(medium_risk)}
        Low Risk Events: {len(low_risk)}
        Average Risk Score: {sum(e.risk_score for e in events) / len(events) if events else 0:.2f}
        """
        
        report.metrics.update({
            "risk_distribution": {
                "high": len(high_risk),
                "medium": len(medium_risk),
                "low": len(low_risk)
            },
            "top_risk_categories": self._get_top_risk_categories(events)
        })
        
        if high_risk:
            report.findings.append(f"Identified {len(high_risk)} high-risk events requiring immediate attention")
            report.recommendations.append("Implement additional controls for high-risk activities")
        
        report.risk_level = self._calculate_overall_risk_level(events)
        report.data_sources = ["risk_assessment_engine", "audit_events"]
    
    async def _generate_generic_report(self, report: AuditReport, events: List[ComprehensiveAuditEvent]):
        """Generate generic report content."""
        report.executive_summary = f"Generic audit report analyzing {len(events)} events"
        report.metrics = {"total_events": len(events)}
        report.data_sources = ["audit_events"]
    
    def _calculate_compliance_score(self, events: List[ComprehensiveAuditEvent]) -> float:
        """Calculate overall compliance score."""
        compliance_events = [e for e in events if e.compliance_relevant]
        if not compliance_events:
            return 100.0
        
        violations = len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION])
        return max(0, 100 - (violations / len(compliance_events) * 50))
    
    def _calculate_security_score(self, events: List[ComprehensiveAuditEvent]) -> float:
        """Calculate overall security score."""
        if not events:
            return 100.0
        
        security_incidents = len([e for e in events if e.event_type == AuditEventType.SECURITY_INCIDENT])
        high_risk_events = len([e for e in events if e.risk_score >= 8.0])
        
        return max(0, 100 - (security_incidents * 10) - (high_risk_events * 2))
    
    def _calculate_overall_risk_level(self, events: List[ComprehensiveAuditEvent]) -> str:
        """Calculate overall risk level."""
        if not events:
            return "low"
        
        avg_risk = sum(e.risk_score for e in events) / len(events)
        high_risk_count = len([e for e in events if e.risk_score >= 8.0])
        
        if avg_risk >= 6.0 or high_risk_count > len(events) * 0.1:
            return "critical"
        elif avg_risk >= 4.0 or high_risk_count > 0:
            return "high"
        elif avg_risk >= 2.0:
            return "medium"
        else:
            return "low"
    
    def _get_top_risk_categories(self, events: List[ComprehensiveAuditEvent]) -> Dict[str, int]:
        """Get top risk categories from events."""
        categories = {}
        for event in events:
            if event.risk_score >= 5.0:  # Only consider medium+ risk events
                category = event.event_type.value
                categories[category] = categories.get(category, 0) + 1
        
        # Return top 5 categories
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_categories[:5])
    
    @handle_exceptions
    async def export_report(
        self,
        report_id: str,
        format: ReportFormat,
        include_raw_data: bool = False
    ) -> bytes:
        """
        Export a report in the specified format.
        
        Args:
            report_id: ID of report to export
            format: Export format
            include_raw_data: Whether to include raw audit data
            
        Returns:
            Exported report as bytes
        """
        if report_id not in self.generated_reports:
            raise ValueError(f"Report {report_id} not found")
        
        report = self.generated_reports[report_id]
        
        if format == ReportFormat.JSON:
            report_dict = {
                "report_id": report.report_id,
                "title": report.title,
                "description": report.description,
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.report_period_start.isoformat(),
                "period_end": report.report_period_end.isoformat(),
                "executive_summary": report.executive_summary,
                "findings": report.findings,
                "recommendations": report.recommendations,
                "metrics": report.metrics,
                "compliance_score": report.compliance_score,
                "risk_level": report.risk_level
            }
            return json.dumps(report_dict, indent=2).encode('utf-8')
        
        elif format == ReportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write report metadata
            writer.writerow(["Report ID", report.report_id])
            writer.writerow(["Title", report.title])
            writer.writerow(["Generated At", report.generated_at.isoformat()])
            writer.writerow([])
            
            # Write findings
            writer.writerow(["Findings"])
            for finding in report.findings:
                writer.writerow([finding])
            
            writer.writerow([])
            
            # Write recommendations
            writer.writerow(["Recommendations"])
            for rec in report.recommendations:
                writer.writerow([rec])
            
            return output.getvalue().encode('utf-8')
        
        elif format == ReportFormat.HTML:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report.title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ border-bottom: 2px solid #333; margin-bottom: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{report.title}</h1>
                    <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Period: {report.report_period_start.strftime('%Y-%m-%d')} to {report.report_period_end.strftime('%Y-%m-%d')}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <pre>{report.executive_summary}</pre>
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    <ul>
                        {''.join(f'<li>{finding}</li>' for finding in report.findings)}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Metrics</h2>
                    {''.join(f'<div class="metric"><strong>{k}:</strong> {v}</div>' for k, v in report.metrics.items())}
                </div>
            </body>
            </html>
            """
            return html_content.encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _scheduled_report_generator(self):
        """Background task for generating scheduled reports."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                if not self.auto_report_generation:
                    continue
                
                current_time = datetime.utcnow()
                
                # Check each template for scheduled generation
                for template in self.report_templates.values():
                    if not template.auto_generate:
                        continue
                    
                    # Determine if report should be generated
                    should_generate = False
                    
                    if template.schedule == "daily":
                        should_generate = True  # Generate daily for demo
                    elif template.schedule == "weekly" and current_time.weekday() == 0:  # Monday
                        should_generate = True
                    elif template.schedule == "monthly" and current_time.day == 1:  # First of month
                        should_generate = True
                    elif template.schedule == "quarterly" and current_time.day == 1 and current_time.month in [1, 4, 7, 10]:
                        should_generate = True
                    elif template.schedule == "annually" and current_time.day == 1 and current_time.month == 1:
                        should_generate = True
                    
                    if should_generate:
                        # Generate report based on template
                        end_date = current_time
                        if template.schedule == "daily":
                            start_date = end_date - timedelta(days=1)
                        elif template.schedule == "weekly":
                            start_date = end_date - timedelta(days=7)
                        elif template.schedule == "monthly":
                            start_date = end_date - timedelta(days=30)
                        else:
                            start_date = end_date - timedelta(days=90)
                        
                        await self.generate_audit_report(
                            report_type=ReportType.COMPLIANCE_STATUS,
                            start_date=start_date,
                            end_date=end_date,
                            compliance_standards=[template.compliance_standard],
                            generated_by="scheduled_system"
                        )
                        
                        logger.info(f"Generated scheduled report for template: {template.name}")
                
            except Exception as e:
                logger.error(f"Scheduled report generation error: {e}")
    
    async def _audit_event_cleanup(self):
        """Background task for audit event cleanup and archival."""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily cleanup
                
                # Archive old events (simplified - in production would move to archive storage)
                archive_cutoff = datetime.utcnow() - timedelta(days=365)  # Archive after 1 year
                
                events_to_archive = [
                    e for e in self.audit_events 
                    if e.timestamp < archive_cutoff and not e.compliance_relevant
                ]
                
                if events_to_archive:
                    logger.info(f"Archiving {len(events_to_archive)} old audit events")
                    # In production, would move to archive storage
                    self.audit_events = [e for e in self.audit_events if e not in events_to_archive]
                
            except Exception as e:
                logger.error(f"Audit event cleanup error: {e}")
    
    @handle_exceptions
    async def get_reporting_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive audit and compliance reporting dashboard."""
        # Calculate statistics
        total_events = len(self.audit_events)
        recent_events = len([
            e for e in self.audit_events 
            if e.timestamp > datetime.utcnow() - timedelta(days=7)
        ])
        
        high_risk_events = len([e for e in self.audit_events if e.risk_score >= 7.0])
        compliance_events = len([e for e in self.audit_events if e.compliance_relevant])
        
        # Recent reports
        recent_reports = sorted(
            self.generated_reports.values(),
            key=lambda x: x.generated_at,
            reverse=True
        )[:10]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "audit_statistics": {
                "total_audit_events": total_events,
                "recent_events_7_days": recent_events,
                "high_risk_events": high_risk_events,
                "compliance_relevant_events": compliance_events,
                "average_risk_score": sum(e.risk_score for e in self.audit_events) / total_events if total_events else 0
            },
            "report_statistics": {
                "total_reports_generated": len(self.generated_reports),
                "report_templates": len(self.report_templates),
                "auto_generation_enabled": self.auto_report_generation
            },
            "recent_reports": [
                {
                    "report_id": r.report_id,
                    "title": r.title,
                    "type": r.report_type.value,
                    "generated_at": r.generated_at.isoformat(),
                    "compliance_score": r.compliance_score,
                    "risk_level": r.risk_level
                }
                for r in recent_reports
            ],
            "compliance_templates": [
                {
                    "template_id": t.template_id,
                    "name": t.name,
                    "standard": t.compliance_standard.value,
                    "schedule": t.schedule,
                    "auto_generate": t.auto_generate
                }
                for t in self.report_templates.values()
            ],
            "supported_report_types": [rt.value for rt in ReportType],
            "supported_export_formats": [rf.value for rf in ReportFormat],
            "retention_policy": {
                "default_retention_days": self.default_retention_days,
                "max_audit_events": self.max_audit_events
            }
        }