"""Comprehensive compliance audit framework for data protection and security standards."""

from __future__ import annotations

import json
import os
import re
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, validator

from ....infrastructure.logging import get_logger

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ....infrastructure.monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"                    # General Data Protection Regulation
    CCPA = "ccpa"                    # California Consumer Privacy Act
    HIPAA = "hipaa"                  # Health Insurance Portability and Accountability Act
    SOX = "sox"                      # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"             # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"         # Information Security Management
    NIST_CSF = "nist_csf"           # NIST Cybersecurity Framework
    SOC2 = "soc2"                   # Service Organization Control 2
    COPPA = "coppa"                 # Children's Online Privacy Protection Act
    FERPA = "ferpa"                 # Family Educational Rights and Privacy Act


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class AuditSeverity(Enum):
    """Audit finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""
    id: str
    standard: ComplianceStandard
    title: str
    description: str
    category: str
    mandatory: bool = True
    evidence_required: List[str] = field(default_factory=list)
    remediation_guidance: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class AuditFinding:
    """Individual audit finding."""
    id: str
    requirement_id: str
    title: str
    description: str
    status: ComplianceStatus
    severity: AuditSeverity
    category: str
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100
    confidence: float = 1.0  # 0-1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    assignee: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    """Compliance audit result."""
    audit_id: str
    standard: ComplianceStandard
    start_time: datetime
    end_time: datetime
    scope: str
    findings: List[AuditFinding]
    total_requirements: int
    compliant_requirements: int
    non_compliant_requirements: int
    compliance_score: float  # 0-100
    overall_status: ComplianceStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def compliance_rate(self) -> float:
        """Calculate compliance rate."""
        if self.total_requirements == 0:
            return 0.0
        return self.compliant_requirements / self.total_requirements * 100

    @property
    def critical_findings(self) -> List[AuditFinding]:
        """Get critical severity findings."""
        return [f for f in self.findings if f.severity == AuditSeverity.CRITICAL]

    @property
    def high_findings(self) -> List[AuditFinding]:
        """Get high severity findings."""
        return [f for f in self.findings if f.severity == AuditSeverity.HIGH]


class BaseComplianceChecker(ABC):
    """Base class for compliance checkers."""

    def __init__(self, standard: ComplianceStandard, config: Optional[Dict[str, Any]] = None):
        self.standard = standard
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.requirements = self._load_requirements()

    @abstractmethod
    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements for the standard."""
        pass

    @abstractmethod
    async def check_compliance(self, scope: str, **kwargs) -> List[AuditFinding]:
        """Check compliance against the standard."""
        pass

    def generate_finding_id(self, requirement_id: str, component: str) -> str:
        """Generate unique finding ID."""
        timestamp = str(int(time.time()))
        content = f"{self.standard.value}_{requirement_id}_{component}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class GDPRComplianceChecker(BaseComplianceChecker):
    """GDPR compliance checker."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ComplianceStandard.GDPR, config)

    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load GDPR requirements."""
        return [
            ComplianceRequirement(
                id="GDPR-01",
                standard=ComplianceStandard.GDPR,
                title="Data Subject Rights",
                description="Implement mechanisms for data subject rights (access, rectification, erasure, portability)",
                category="Data Subject Rights",
                evidence_required=["privacy_policy", "data_subject_request_process", "api_endpoints"],
                remediation_guidance="Implement API endpoints and processes for handling data subject requests",
                references=["Article 15-22 GDPR"]
            ),
            ComplianceRequirement(
                id="GDPR-02",
                standard=ComplianceStandard.GDPR,
                title="Consent Management",
                description="Implement lawful basis and consent management for data processing",
                category="Lawful Basis",
                evidence_required=["consent_records", "privacy_notices", "opt_in_mechanisms"],
                remediation_guidance="Implement consent management system with clear opt-in/out mechanisms",
                references=["Article 6, 7 GDPR"]
            ),
            ComplianceRequirement(
                id="GDPR-03",
                standard=ComplianceStandard.GDPR,
                title="Data Protection by Design",
                description="Implement privacy by design and default principles",
                category="Technical Measures",
                evidence_required=["system_design", "privacy_impact_assessment", "default_settings"],
                remediation_guidance="Integrate privacy considerations into system design and default configurations",
                references=["Article 25 GDPR"]
            ),
            ComplianceRequirement(
                id="GDPR-04",
                standard=ComplianceStandard.GDPR,
                title="Data Breach Notification",
                description="Implement data breach detection and notification procedures",
                category="Incident Response",
                evidence_required=["incident_response_plan", "breach_detection", "notification_procedures"],
                remediation_guidance="Establish 72-hour breach notification process to supervisory authorities",
                references=["Article 33, 34 GDPR"]
            ),
            ComplianceRequirement(
                id="GDPR-05",
                standard=ComplianceStandard.GDPR,
                title="Data Minimization",
                description="Ensure data processing is limited to what is necessary",
                category="Data Processing",
                evidence_required=["data_inventory", "retention_policies", "data_classification"],
                remediation_guidance="Implement data minimization practices and regular data purging",
                references=["Article 5(1)(c) GDPR"]
            )
        ]

    async def check_compliance(self, scope: str, **kwargs) -> List[AuditFinding]:
        """Check GDPR compliance."""
        findings = []
        
        # Check data subject rights implementation
        findings.extend(await self._check_data_subject_rights(scope))
        
        # Check consent management
        findings.extend(await self._check_consent_management(scope))
        
        # Check privacy by design
        findings.extend(await self._check_privacy_by_design(scope))
        
        # Check breach notification readiness
        findings.extend(await self._check_breach_notification(scope))
        
        # Check data minimization
        findings.extend(await self._check_data_minimization(scope))
        
        return findings

    async def _check_data_subject_rights(self, scope: str) -> List[AuditFinding]:
        """Check implementation of data subject rights."""
        findings = []
        
        # Check for API endpoints
        api_endpoints = [
            "/api/v1/privacy/data-access",
            "/api/v1/privacy/data-export",
            "/api/v1/privacy/data-deletion",
            "/api/v1/privacy/data-rectification"
        ]
        
        missing_endpoints = []
        for endpoint in api_endpoints:
            if not await self._check_endpoint_exists(scope, endpoint):
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-01", "api_endpoints"),
                requirement_id="GDPR-01",
                title="Missing Data Subject Rights API Endpoints",
                description=f"Missing API endpoints for data subject rights: {', '.join(missing_endpoints)}",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Data Subject Rights",
                evidence=f"Missing endpoints: {missing_endpoints}",
                remediation="Implement missing API endpoints for data subject rights",
                affected_systems=[scope],
                risk_score=80.0
            )
            findings.append(finding)
        
        # Check for privacy policy
        if not await self._check_privacy_policy_exists(scope):
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-01", "privacy_policy"),
                requirement_id="GDPR-01",
                title="Missing Privacy Policy",
                description="No privacy policy found",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.CRITICAL,
                category="Data Subject Rights",
                evidence="Privacy policy document not found",
                remediation="Create and publish comprehensive privacy policy",
                affected_systems=[scope],
                risk_score=90.0
            )
            findings.append(finding)
        
        return findings

    async def _check_consent_management(self, scope: str) -> List[AuditFinding]:
        """Check consent management implementation."""
        findings = []
        
        # Check for consent tracking
        if not await self._check_consent_tracking(scope):
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-02", "consent_tracking"),
                requirement_id="GDPR-02",
                title="Missing Consent Tracking",
                description="No consent tracking mechanism found",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Lawful Basis",
                evidence="Consent tracking system not implemented",
                remediation="Implement consent tracking and management system",
                affected_systems=[scope],
                risk_score=75.0
            )
            findings.append(finding)
        
        return findings

    async def _check_privacy_by_design(self, scope: str) -> List[AuditFinding]:
        """Check privacy by design implementation."""
        findings = []
        
        # Check default privacy settings
        if not await self._check_privacy_defaults(scope):
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-03", "privacy_defaults"),
                requirement_id="GDPR-03",
                title="Privacy by Default Not Implemented",
                description="System does not implement privacy by default settings",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.MEDIUM,
                category="Technical Measures",
                evidence="Default settings favor data collection over privacy",
                remediation="Configure system defaults to maximize privacy protection",
                affected_systems=[scope],
                risk_score=60.0
            )
            findings.append(finding)
        
        return findings

    async def _check_breach_notification(self, scope: str) -> List[AuditFinding]:
        """Check breach notification readiness."""
        findings = []
        
        # Check for incident response plan
        if not await self._check_incident_response_plan(scope):
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-04", "incident_response"),
                requirement_id="GDPR-04",
                title="Missing Incident Response Plan",
                description="No documented incident response plan for data breaches",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Incident Response",
                evidence="Incident response plan documentation not found",
                remediation="Develop and implement incident response plan with 72-hour notification process",
                affected_systems=[scope],
                risk_score=85.0
            )
            findings.append(finding)
        
        return findings

    async def _check_data_minimization(self, scope: str) -> List[AuditFinding]:
        """Check data minimization practices."""
        findings = []
        
        # Check for data retention policies
        if not await self._check_retention_policies(scope):
            finding = AuditFinding(
                id=self.generate_finding_id("GDPR-05", "retention_policies"),
                requirement_id="GDPR-05",
                title="Missing Data Retention Policies",
                description="No documented data retention and deletion policies",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.MEDIUM,
                category="Data Processing",
                evidence="Data retention policies not documented or implemented",
                remediation="Implement data retention policies with automated deletion",
                affected_systems=[scope],
                risk_score=65.0
            )
            findings.append(finding)
        
        return findings

    # Helper methods for checks
    async def _check_endpoint_exists(self, scope: str, endpoint: str) -> bool:
        """Check if API endpoint exists."""
        # This would integrate with API discovery/testing
        # For now, simulate checks
        return False  # Assume missing for demonstration

    async def _check_privacy_policy_exists(self, scope: str) -> bool:
        """Check if privacy policy exists."""
        policy_paths = [
            "docs/privacy-policy.md",
            "legal/privacy-policy.html",
            "static/privacy.html"
        ]
        
        for path in policy_paths:
            if Path(scope) / path:
                if (Path(scope) / path).exists():
                    return True
        return False

    async def _check_consent_tracking(self, scope: str) -> bool:
        """Check consent tracking implementation."""
        # Check for consent-related code patterns
        consent_patterns = [
            r'consent[_-]?tracking',
            r'user[_-]?consent',
            r'privacy[_-]?settings',
            r'opt[_-]?in'
        ]
        
        return await self._check_code_patterns(scope, consent_patterns)

    async def _check_privacy_defaults(self, scope: str) -> bool:
        """Check privacy by default settings."""
        privacy_patterns = [
            r'privacy[_-]?by[_-]?default',
            r'default[_-]?privacy',
            r'minimize[_-]?data',
            r'opt[_-]?out[_-]?default'
        ]
        
        return await self._check_code_patterns(scope, privacy_patterns)

    async def _check_incident_response_plan(self, scope: str) -> bool:
        """Check for incident response plan."""
        plan_paths = [
            "docs/incident-response.md",
            "security/incident-response-plan.md",
            "docs/security/breach-response.md"
        ]
        
        for path in plan_paths:
            if (Path(scope) / path).exists():
                return True
        return False

    async def _check_retention_policies(self, scope: str) -> bool:
        """Check for data retention policies."""
        retention_patterns = [
            r'data[_-]?retention',
            r'retention[_-]?policy',
            r'data[_-]?purge',
            r'auto[_-]?delete'
        ]
        
        return await self._check_code_patterns(scope, retention_patterns)

    async def _check_code_patterns(self, scope: str, patterns: List[str]) -> bool:
        """Check for code patterns in the scope."""
        try:
            for root, dirs, files in os.walk(scope):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.cs')):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                for pattern in patterns:
                                    if re.search(pattern, content, re.IGNORECASE):
                                        return True
                        except (UnicodeDecodeError, IOError):
                            continue
        except Exception as e:
            logger.debug(f"Error checking code patterns: {e}")
        
        return False


class HIPAAComplianceChecker(BaseComplianceChecker):
    """HIPAA compliance checker."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ComplianceStandard.HIPAA, config)

    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load HIPAA requirements."""
        return [
            ComplianceRequirement(
                id="HIPAA-01",
                standard=ComplianceStandard.HIPAA,
                title="Administrative Safeguards",
                description="Implement administrative safeguards for PHI protection",
                category="Administrative",
                evidence_required=["policies", "training_records", "access_controls"],
                remediation_guidance="Establish administrative policies and training programs",
                references=["164.308 HIPAA"]
            ),
            ComplianceRequirement(
                id="HIPAA-02",
                standard=ComplianceStandard.HIPAA,
                title="Physical Safeguards",
                description="Implement physical safeguards for systems containing PHI",
                category="Physical",
                evidence_required=["facility_access_controls", "workstation_controls", "device_controls"],
                remediation_guidance="Implement physical access controls and device management",
                references=["164.310 HIPAA"]
            ),
            ComplianceRequirement(
                id="HIPAA-03",
                standard=ComplianceStandard.HIPAA,
                title="Technical Safeguards",
                description="Implement technical safeguards for PHI transmission and storage",
                category="Technical",
                evidence_required=["encryption", "access_controls", "audit_logs", "data_integrity"],
                remediation_guidance="Implement encryption, access controls, and audit logging",
                references=["164.312 HIPAA"]
            )
        ]

    async def check_compliance(self, scope: str, **kwargs) -> List[AuditFinding]:
        """Check HIPAA compliance."""
        findings = []
        
        # Check technical safeguards
        findings.extend(await self._check_technical_safeguards(scope))
        
        # Check encryption requirements
        findings.extend(await self._check_encryption_requirements(scope))
        
        # Check audit logging
        findings.extend(await self._check_audit_logging(scope))
        
        return findings

    async def _check_technical_safeguards(self, scope: str) -> List[AuditFinding]:
        """Check technical safeguards implementation."""
        findings = []
        
        # Check for encryption patterns
        encryption_patterns = [
            r'encrypt',
            r'aes[_-]?256',
            r'tls[_-]?1\.[23]',
            r'ssl[_-]?context'
        ]
        
        if not await self._check_code_patterns(scope, encryption_patterns):
            finding = AuditFinding(
                id=self.generate_finding_id("HIPAA-03", "encryption"),
                requirement_id="HIPAA-03",
                title="Missing Encryption Implementation",
                description="No evidence of encryption implementation found",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.CRITICAL,
                category="Technical",
                evidence="Encryption patterns not found in codebase",
                remediation="Implement encryption for PHI data at rest and in transit",
                affected_systems=[scope],
                risk_score=95.0
            )
            findings.append(finding)
        
        return findings

    async def _check_encryption_requirements(self, scope: str) -> List[AuditFinding]:
        """Check encryption requirements."""
        findings = []
        
        # Check for weak encryption patterns
        weak_patterns = [
            r'md5',
            r'sha1(?![0-9])',
            r'des[^c]',
            r'rc4'
        ]
        
        if await self._check_code_patterns(scope, weak_patterns):
            finding = AuditFinding(
                id=self.generate_finding_id("HIPAA-03", "weak_encryption"),
                requirement_id="HIPAA-03",
                title="Weak Encryption Algorithms Detected",
                description="Use of weak or deprecated encryption algorithms",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Technical",
                evidence="Weak encryption patterns found in codebase",
                remediation="Replace weak encryption with approved algorithms (AES-256, SHA-256+)",
                affected_systems=[scope],
                risk_score=85.0
            )
            findings.append(finding)
        
        return findings

    async def _check_audit_logging(self, scope: str) -> List[AuditFinding]:
        """Check audit logging implementation."""
        findings = []
        
        # Check for audit logging patterns
        audit_patterns = [
            r'audit[_-]?log',
            r'security[_-]?log',
            r'access[_-]?log',
            r'logging\.getLogger'
        ]
        
        if not await self._check_code_patterns(scope, audit_patterns):
            finding = AuditFinding(
                id=self.generate_finding_id("HIPAA-03", "audit_logging"),
                requirement_id="HIPAA-03",
                title="Insufficient Audit Logging",
                description="No evidence of comprehensive audit logging found",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Technical",
                evidence="Audit logging patterns not found in codebase",
                remediation="Implement comprehensive audit logging for all PHI access",
                affected_systems=[scope],
                risk_score=80.0
            )
            findings.append(finding)
        
        return findings

    async def _check_code_patterns(self, scope: str, patterns: List[str]) -> bool:
        """Check for code patterns in the scope."""
        try:
            for root, dirs, files in os.walk(scope):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.cs')):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                for pattern in patterns:
                                    if re.search(pattern, content, re.IGNORECASE):
                                        return True
                        except (UnicodeDecodeError, IOError):
                            continue
        except Exception as e:
            logger.debug(f"Error checking code patterns: {e}")
        
        return False


class ComplianceAuditor:
    """Main compliance auditor orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance auditor.
        
        Args:
            config: Auditor configuration dictionary
        """
        self.config = config or {}
        
        # Initialize checkers
        self.checkers = {
            ComplianceStandard.GDPR: GDPRComplianceChecker(self.config.get("gdpr", {})),
            ComplianceStandard.HIPAA: HIPAAComplianceChecker(self.config.get("hipaa", {}))
        }
        
        # Filter enabled checkers
        self.enabled_checkers = {
            standard: checker for standard, checker in self.checkers.items()
            if checker.enabled
        }
        
        logger.info("ComplianceAuditor initialized",
                   enabled_standards=[s.value for s in self.enabled_checkers.keys()])

    async def audit_all_standards(self, scope: str) -> List[AuditResult]:
        """Run compliance audit for all enabled standards.
        
        Args:
            scope: Audit scope (directory path, system identifier, etc.)
            
        Returns:
            List of audit results for each standard
        """
        results = []
        
        for standard, checker in self.enabled_checkers.items():
            try:
                result = await self.audit_standard(standard, scope)
                results.append(result)
            except Exception as e:
                logger.error(f"Audit failed for {standard.value}",
                           error=str(e), scope=scope)
        
        return results

    async def audit_standard(
        self,
        standard: ComplianceStandard,
        scope: str,
        **kwargs
    ) -> AuditResult:
        """Run compliance audit for specific standard.
        
        Args:
            standard: Compliance standard to audit
            scope: Audit scope
            **kwargs: Additional parameters for the checker
            
        Returns:
            Audit result for the standard
        """
        if standard not in self.enabled_checkers:
            raise ValueError(f"Standard {standard.value} not enabled or not supported")
        
        audit_id = self._generate_audit_id(standard)
        start_time = datetime.utcnow()
        
        logger.info("Starting compliance audit",
                   audit_id=audit_id,
                   standard=standard.value,
                   scope=scope)
        
        checker = self.enabled_checkers[standard]
        
        try:
            # Run compliance checks
            findings = await checker.check_compliance(scope, **kwargs)
            
            end_time = datetime.utcnow()
            
            # Calculate compliance metrics
            total_requirements = len(checker.requirements)
            compliant_count = len([f for f in findings if f.status == ComplianceStatus.COMPLIANT])
            non_compliant_count = len([f for f in findings if f.status == ComplianceStatus.NON_COMPLIANT])
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(findings, checker.requirements)
            
            # Determine overall status
            overall_status = self._determine_overall_status(findings)
            
            result = AuditResult(
                audit_id=audit_id,
                standard=standard,
                start_time=start_time,
                end_time=end_time,
                scope=scope,
                findings=findings,
                total_requirements=total_requirements,
                compliant_requirements=compliant_count,
                non_compliant_requirements=non_compliant_count,
                compliance_score=compliance_score,
                overall_status=overall_status,
                metadata={
                    "checker_version": "1.0",
                    "audit_duration": (end_time - start_time).total_seconds()
                }
            )
            
            # Record metrics
            metrics = get_safe_metrics_collector()
            metrics.record_metric(
                f"compliance.audit.{standard.value}.score",
                compliance_score,
                {"audit_id": audit_id}
            )
            
            metrics.record_metric(
                f"compliance.audit.{standard.value}.findings",
                len(findings),
                {"audit_id": audit_id}
            )
            
            logger.info("Compliance audit completed",
                       audit_id=audit_id,
                       standard=standard.value,
                       compliance_score=compliance_score,
                       findings_count=len(findings),
                       critical_findings=len(result.critical_findings))
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            logger.error("Compliance audit failed",
                        audit_id=audit_id,
                        standard=standard.value,
                        error=str(e))
            
            # Return empty result with error metadata
            return AuditResult(
                audit_id=audit_id,
                standard=standard,
                start_time=start_time,
                end_time=end_time,
                scope=scope,
                findings=[],
                total_requirements=len(checker.requirements),
                compliant_requirements=0,
                non_compliant_requirements=0,
                compliance_score=0.0,
                overall_status=ComplianceStatus.REQUIRES_REVIEW,
                metadata={"error": str(e)}
            )

    def generate_compliance_report(
        self,
        audit_results: List[AuditResult],
        format: str = "json"
    ) -> Union[Dict[str, Any], str]:
        """Generate comprehensive compliance report.
        
        Args:
            audit_results: List of audit results
            format: Report format ("json", "html", "csv")
            
        Returns:
            Compliance report in requested format
        """
        if format == "json":
            return self._generate_json_report(audit_results)
        elif format == "html":
            return self._generate_html_report(audit_results)
        elif format == "csv":
            return self._generate_csv_report(audit_results)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_json_report(self, audit_results: List[AuditResult]) -> Dict[str, Any]:
        """Generate JSON compliance report."""
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "standards_audited": [r.standard.value for r in audit_results],
                "total_audits": len(audit_results)
            },
            "executive_summary": {
                "overall_compliance_score": self._calculate_overall_score(audit_results),
                "total_findings": sum(len(r.findings) for r in audit_results),
                "critical_findings": sum(len(r.critical_findings) for r in audit_results),
                "high_findings": sum(len(r.high_findings) for r in audit_results),
                "standards_summary": {}
            },
            "detailed_results": []
        }
        
        # Add standards summary
        for result in audit_results:
            report["executive_summary"]["standards_summary"][result.standard.value] = {
                "compliance_score": result.compliance_score,
                "overall_status": result.overall_status.value,
                "total_findings": len(result.findings),
                "critical_findings": len(result.critical_findings),
                "high_findings": len(result.high_findings)
            }
        
        # Add detailed results
        for result in audit_results:
            detailed_result = {
                "audit_id": result.audit_id,
                "standard": result.standard.value,
                "scope": result.scope,
                "audit_period": {
                    "start": result.start_time.isoformat(),
                    "end": result.end_time.isoformat()
                },
                "compliance_metrics": {
                    "score": result.compliance_score,
                    "status": result.overall_status.value,
                    "compliance_rate": result.compliance_rate,
                    "total_requirements": result.total_requirements,
                    "compliant_requirements": result.compliant_requirements,
                    "non_compliant_requirements": result.non_compliant_requirements
                },
                "findings": [
                    {
                        "id": f.id,
                        "requirement_id": f.requirement_id,
                        "title": f.title,
                        "description": f.description,
                        "status": f.status.value,
                        "severity": f.severity.value,
                        "category": f.category,
                        "risk_score": f.risk_score,
                        "confidence": f.confidence,
                        "evidence": f.evidence,
                        "remediation": f.remediation,
                        "affected_systems": f.affected_systems,
                        "timestamp": f.timestamp.isoformat()
                    }
                    for f in result.findings
                ],
                "recommendations": self._generate_recommendations(result)
            }
            report["detailed_results"].append(detailed_result)
        
        return report

    def _generate_html_report(self, audit_results: List[AuditResult]) -> str:
        """Generate HTML compliance report."""
        # This would generate a comprehensive HTML report
        # For brevity, returning a simple template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Audit Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f5f5f5; padding: 20px; }
                .summary { margin: 20px 0; }
                .finding { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
                .critical { border-left: 5px solid #d32f2f; }
                .high { border-left: 5px solid #f57c00; }
                .medium { border-left: 5px solid #fbc02d; }
                .low { border-left: 5px solid #388e3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Audit Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>Overall Compliance Score: {overall_score}%</p>
                <p>Total Findings: {total_findings}</p>
                <p>Critical Findings: {critical_findings}</p>
            </div>
            
            <!-- Detailed findings would be rendered here -->
        </body>
        </html>
        """.format(
            timestamp=datetime.utcnow().isoformat(),
            overall_score=self._calculate_overall_score(audit_results),
            total_findings=sum(len(r.findings) for r in audit_results),
            critical_findings=sum(len(r.critical_findings) for r in audit_results)
        )
        
        return html_template

    def _generate_csv_report(self, audit_results: List[AuditResult]) -> str:
        """Generate CSV compliance report."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Audit ID", "Standard", "Finding ID", "Requirement ID", "Title",
            "Status", "Severity", "Category", "Risk Score", "Confidence",
            "Affected Systems", "Timestamp"
        ])
        
        # Write findings
        for result in audit_results:
            for finding in result.findings:
                writer.writerow([
                    result.audit_id,
                    result.standard.value,
                    finding.id,
                    finding.requirement_id,
                    finding.title,
                    finding.status.value,
                    finding.severity.value,
                    finding.category,
                    finding.risk_score,
                    finding.confidence,
                    "|".join(finding.affected_systems),
                    finding.timestamp.isoformat()
                ])
        
        return output.getvalue()

    def _generate_audit_id(self, standard: ComplianceStandard) -> str:
        """Generate unique audit ID."""
        timestamp = str(int(time.time()))
        return f"audit_{standard.value}_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"

    def _calculate_compliance_score(
        self,
        findings: List[AuditFinding],
        requirements: List[ComplianceRequirement]
    ) -> float:
        """Calculate compliance score (0-100)."""
        if not requirements:
            return 0.0
        
        # Weight findings by severity
        severity_weights = {
            AuditSeverity.CRITICAL: -30,
            AuditSeverity.HIGH: -20,
            AuditSeverity.MEDIUM: -10,
            AuditSeverity.LOW: -5,
            AuditSeverity.INFO: -1
        }
        
        total_deductions = 0
        for finding in findings:
            if finding.status == ComplianceStatus.NON_COMPLIANT:
                weight = severity_weights.get(finding.severity, -10)
                total_deductions += weight * finding.confidence
        
        # Start with perfect score and deduct
        base_score = 100.0
        final_score = base_score + total_deductions  # deductions are negative
        
        return max(0.0, min(100.0, final_score))

    def _determine_overall_status(self, findings: List[AuditFinding]) -> ComplianceStatus:
        """Determine overall compliance status."""
        if not findings:
            return ComplianceStatus.COMPLIANT
        
        # Check for critical or high severity non-compliant findings
        critical_high_findings = [
            f for f in findings
            if f.status == ComplianceStatus.NON_COMPLIANT and
            f.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]
        ]
        
        if critical_high_findings:
            return ComplianceStatus.NON_COMPLIANT
        
        # Check for any non-compliant findings
        non_compliant_findings = [
            f for f in findings if f.status == ComplianceStatus.NON_COMPLIANT
        ]
        
        if non_compliant_findings:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        
        return ComplianceStatus.COMPLIANT

    def _calculate_overall_score(self, audit_results: List[AuditResult]) -> float:
        """Calculate overall compliance score across all standards."""
        if not audit_results:
            return 0.0
        
        return sum(r.compliance_score for r in audit_results) / len(audit_results)

    def _generate_recommendations(self, audit_result: AuditResult) -> List[str]:
        """Generate recommendations based on audit findings."""
        recommendations = []
        
        # Priority recommendations for critical/high findings
        critical_high_findings = [
            f for f in audit_result.findings
            if f.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]
        ]
        
        if critical_high_findings:
            recommendations.append(
                f"Immediately address {len(critical_high_findings)} critical/high priority compliance gaps"
            )
        
        # Category-specific recommendations
        categories = set(f.category for f in audit_result.findings if f.status == ComplianceStatus.NON_COMPLIANT)
        
        if "Technical" in categories:
            recommendations.append("Implement technical safeguards including encryption and access controls")
        
        if "Data Subject Rights" in categories:
            recommendations.append("Develop data subject request handling processes and API endpoints")
        
        if "Incident Response" in categories:
            recommendations.append("Create comprehensive incident response and breach notification procedures")
        
        return recommendations


# Global auditor instance
_compliance_auditor: Optional[ComplianceAuditor] = None


def get_compliance_auditor(config: Optional[Dict[str, Any]] = None) -> ComplianceAuditor:
    """Get the global compliance auditor instance."""
    global _compliance_auditor
    
    if _compliance_auditor is None or config is not None:
        _compliance_auditor = ComplianceAuditor(config)
    
    return _compliance_auditor


def initialize_compliance_auditor(config: Optional[Dict[str, Any]] = None) -> ComplianceAuditor:
    """Initialize the global compliance auditor."""
    global _compliance_auditor
    _compliance_auditor = ComplianceAuditor(config)
    return _compliance_auditor