"""
Regulatory Compliance Framework

Comprehensive regulatory compliance management for ML systems,
supporting multiple regulatory frameworks with automated compliance
checking, reporting, and certification workflows.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import hashlib
import base64

import pandas as pd
from pydantic import BaseModel, Field
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .ml_governance_framework import (
    MLGovernanceFramework, ComplianceFramework, GovernanceRisk,
    AuditEventType, ComplianceCheck
)


class RegulationType(Enum):
    """Types of regulatory requirements."""
    DATA_PROTECTION = "data_protection"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    GOVERNMENT = "government"
    INDUSTRY_SPECIFIC = "industry_specific"
    CROSS_BORDER = "cross_border"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    PENDING_CERTIFICATION = "pending_certification"
    CERTIFIED = "certified"
    EXPIRED = "expired"


class DataClassification(Enum):
    """Data sensitivity classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class RegulatoryRequirement:
    """Individual regulatory requirement definition."""
    requirement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework: ComplianceFramework = ComplianceFramework.GDPR
    regulation_type: RegulationType = RegulationType.DATA_PROTECTION
    
    # Requirement details
    title: str = ""
    description: str = ""
    legal_reference: str = ""  # Article/section reference
    jurisdiction: str = ""  # Geographic jurisdiction
    
    # Implementation requirements
    technical_controls: List[str] = field(default_factory=list)
    administrative_controls: List[str] = field(default_factory=list)
    physical_controls: List[str] = field(default_factory=list)
    
    # Compliance criteria
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    evidence_requirements: List[str] = field(default_factory=list)
    
    # Risk and priority
    risk_level: GovernanceRisk = GovernanceRisk.MEDIUM
    mandatory: bool = True
    deadline: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reviewed: Optional[datetime] = None
    review_frequency_days: int = 365


@dataclass
class ComplianceCertification:
    """Compliance certification record."""
    certification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework: ComplianceFramework = ComplianceFramework.GDPR
    model_id: Optional[str] = None
    system_scope: str = ""
    
    # Certification details
    certification_type: str = ""  # self-assessment, third-party, regulatory
    certifying_authority: str = ""
    certificate_number: str = ""
    
    # Status and validity
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW
    issued_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    # Assessment results
    compliance_score: float = 0.0
    requirements_met: List[str] = field(default_factory=list)
    requirements_failed: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Documentation
    assessment_report_path: Optional[str] = None
    evidence_package_path: Optional[str] = None
    
    # Maintenance
    next_review_date: Optional[datetime] = None
    renewal_required: bool = True


@dataclass
class DataInventory:
    """Data inventory for compliance tracking."""
    inventory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_source: str = ""
    data_type: str = ""
    
    # Classification
    data_classification: DataClassification = DataClassification.INTERNAL
    contains_pii: bool = False
    contains_phi: bool = False
    contains_financial_data: bool = False
    
    # Legal basis and consent
    legal_basis: List[str] = field(default_factory=list)  # GDPR legal basis
    consent_required: bool = False
    consent_obtained: bool = False
    consent_expiry: Optional[datetime] = None
    
    # Processing details
    processing_purposes: List[str] = field(default_factory=list)
    retention_period_days: Optional[int] = None
    cross_border_transfers: List[str] = field(default_factory=list)
    
    # Geographic scope
    data_residency_requirements: List[str] = field(default_factory=list)
    applicable_jurisdictions: List[str] = field(default_factory=list)
    
    # Technical measures
    encryption_at_rest: bool = False
    encryption_in_transit: bool = False
    access_controls: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RegulatoryComplianceManager:
    """Comprehensive regulatory compliance management system."""
    
    def __init__(self, 
                 governance_framework: MLGovernanceFramework,
                 config: Dict[str, Any] = None):
        self.governance = governance_framework
        self.config = config or {}
        self.logger = structlog.get_logger(__name__)
        
        # Regulatory requirements
        self.requirements: Dict[str, RegulatoryRequirement] = {}
        self.requirement_templates = {}
        
        # Certifications and assessments
        self.certifications: Dict[str, ComplianceCertification] = {}
        self.assessment_engine = ComplianceAssessmentEngine()
        
        # Data governance
        self.data_inventory: Dict[str, DataInventory] = {}
        self.privacy_engine = PrivacyComplianceEngine()
        
        # Reporting and documentation
        self.report_generator = ComplianceReportGenerator()
        self.evidence_manager = EvidenceManager()
        
        # Compliance monitoring
        self.compliance_monitor = RegulatoryComplianceMonitor()
        
        # Initialize regulatory templates
        self._initialize_regulatory_templates()
    
    def _initialize_regulatory_templates(self) -> None:
        """Initialize templates for common regulatory frameworks."""
        
        # GDPR Requirements Template
        gdpr_requirements = [
            {
                "title": "Data Subject Consent",
                "description": "Obtain explicit consent for personal data processing",
                "legal_reference": "Article 6, Article 7",
                "technical_controls": ["consent_management_system", "opt_out_mechanism"],
                "validation_criteria": {"consent_recorded": True, "consent_explicit": True}
            },
            {
                "title": "Data Minimization",
                "description": "Process only necessary personal data",
                "legal_reference": "Article 5(1)(c)",
                "technical_controls": ["feature_selection_audit", "data_usage_monitoring"],
                "validation_criteria": {"data_necessity_documented": True}
            },
            {
                "title": "Right to Erasure",
                "description": "Implement data deletion capabilities",
                "legal_reference": "Article 17",
                "technical_controls": ["automated_deletion", "deletion_verification"],
                "validation_criteria": {"deletion_capability": True, "retention_policies": True}
            }
        ]
        
        self.requirement_templates[ComplianceFramework.GDPR] = gdpr_requirements
    
    async def create_regulatory_requirement(self,
                                          framework: ComplianceFramework,
                                          regulation_type: RegulationType,
                                          title: str,
                                          description: str,
                                          legal_reference: str,
                                          **kwargs) -> str:
        """Create a new regulatory requirement."""
        
        requirement = RegulatoryRequirement(
            framework=framework,
            regulation_type=regulation_type,
            title=title,
            description=description,
            legal_reference=legal_reference,
            **kwargs
        )
        
        self.requirements[requirement.requirement_id] = requirement
        
        await self.governance.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id="regulatory_system",
            action="create_regulatory_requirement",
            resource=f"requirement:{requirement.requirement_id}",
            details={
                "framework": framework.value,
                "regulation_type": regulation_type.value,
                "title": title,
                "legal_reference": legal_reference
            }
        )
        
        self.logger.info(
            "Regulatory requirement created",
            requirement_id=requirement.requirement_id,
            framework=framework.value,
            title=title
        )
        
        return requirement.requirement_id
    
    async def assess_regulatory_compliance(self,
                                         model_id: str,
                                         framework: ComplianceFramework,
                                         scope: str = "full_system") -> str:
        """Perform comprehensive regulatory compliance assessment."""
        
        assessment_id = str(uuid.uuid4())
        
        # Get applicable requirements
        applicable_requirements = [
            req for req in self.requirements.values()
            if req.framework == framework
        ]
        
        # Perform assessment
        assessment_results = await self.assessment_engine.assess_compliance(
            model_id, applicable_requirements, self.data_inventory
        )
        
        # Calculate overall compliance score
        compliance_score = await self._calculate_compliance_score(assessment_results)
        
        # Determine compliance status
        status = await self._determine_compliance_status(compliance_score, assessment_results)
        
        # Create certification record
        certification = ComplianceCertification(
            certification_id=assessment_id,
            framework=framework,
            model_id=model_id,
            system_scope=scope,
            certification_type="automated_assessment",
            certifying_authority="internal_compliance_system",
            status=status,
            compliance_score=compliance_score,
            requirements_met=[r["requirement_id"] for r in assessment_results if r["compliant"]],
            requirements_failed=[r["requirement_id"] for r in assessment_results if not r["compliant"]],
            findings=[r for r in assessment_results if r.get("findings")]
        )
        
        # Set expiry date (1 year for most frameworks)
        if status == ComplianceStatus.COMPLIANT:
            certification.issued_date = datetime.utcnow()
            certification.expiry_date = datetime.utcnow() + timedelta(days=365)
            certification.next_review_date = datetime.utcnow() + timedelta(days=90)
        
        self.certifications[assessment_id] = certification
        
        # Generate assessment report
        report_path = await self.report_generator.generate_compliance_report(
            certification, assessment_results
        )
        certification.assessment_report_path = report_path
        
        # Log assessment
        await self.governance.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id="regulatory_system",
            model_id=model_id,
            action="regulatory_compliance_assessment",
            resource=f"certification:{assessment_id}",
            details={
                "framework": framework.value,
                "compliance_score": compliance_score,
                "status": status.value,
                "requirements_assessed": len(applicable_requirements),
                "requirements_met": len(certification.requirements_met),
                "requirements_failed": len(certification.requirements_failed)
            }
        )
        
        self.logger.info(
            "Regulatory compliance assessment completed",
            assessment_id=assessment_id,
            model_id=model_id,
            framework=framework.value,
            compliance_score=compliance_score,
            status=status.value
        )
        
        return assessment_id
    
    async def register_data_inventory(self,
                                    data_source: str,
                                    data_type: str,
                                    classification: DataClassification,
                                    **kwargs) -> str:
        """Register a data source in the compliance inventory."""
        
        inventory = DataInventory(
            data_source=data_source,
            data_type=data_type,
            data_classification=classification,
            **kwargs
        )
        
        self.data_inventory[inventory.inventory_id] = inventory
        
        # Perform privacy compliance check
        privacy_compliance = await self.privacy_engine.assess_privacy_compliance(inventory)
        
        await self.governance.log_audit_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="data_governance_system",
            action="register_data_inventory",
            resource=f"data_inventory:{inventory.inventory_id}",
            details={
                "data_source": data_source,
                "data_type": data_type,
                "classification": classification.value,
                "contains_pii": inventory.contains_pii,
                "privacy_compliant": privacy_compliance.get("compliant", False)
            }
        )
        
        self.logger.info(
            "Data inventory registered",
            inventory_id=inventory.inventory_id,
            data_source=data_source,
            classification=classification.value
        )
        
        return inventory.inventory_id
    
    async def generate_regulatory_report(self,
                                       framework: ComplianceFramework,
                                       report_type: str = "compliance_status",
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive regulatory compliance report."""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=90)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get relevant certifications
        framework_certifications = [
            cert for cert in self.certifications.values()
            if cert.framework == framework and 
            start_date <= cert.created_at <= end_date  # Using created_at as fallback
        ]
        
        # Get applicable requirements
        framework_requirements = [
            req for req in self.requirements.values()
            if req.framework == framework
        ]
        
        # Calculate compliance metrics
        overall_compliance_score = await self._calculate_overall_compliance_score(
            framework_certifications
        )
        
        # Identify compliance gaps
        compliance_gaps = await self._identify_compliance_gaps(
            framework_requirements, framework_certifications
        )
        
        # Generate recommendations
        recommendations = await self._generate_compliance_recommendations(
            framework, compliance_gaps
        )
        
        report = {
            "report_metadata": {
                "framework": framework.value,
                "report_type": report_type,
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            },
            "executive_summary": {
                "overall_compliance_score": overall_compliance_score,
                "total_requirements": len(framework_requirements),
                "compliant_systems": len([c for c in framework_certifications if c.status == ComplianceStatus.COMPLIANT]),
                "non_compliant_systems": len([c for c in framework_certifications if c.status == ComplianceStatus.NON_COMPLIANT]),
                "pending_reviews": len([c for c in framework_certifications if c.status == ComplianceStatus.UNDER_REVIEW])
            },
            "detailed_assessment": {
                "requirements": [
                    {
                        "requirement_id": req.requirement_id,
                        "title": req.title,
                        "compliance_status": await self._get_requirement_compliance_status(req, framework_certifications),
                        "risk_level": req.risk_level.value,
                        "last_assessed": await self._get_requirement_last_assessment(req, framework_certifications)
                    }
                    for req in framework_requirements
                ],
                "certifications": [
                    {
                        "certification_id": cert.certification_id,
                        "model_id": cert.model_id,
                        "status": cert.status.value,
                        "compliance_score": cert.compliance_score,
                        "issued_date": cert.issued_date.isoformat() if cert.issued_date else None,
                        "expiry_date": cert.expiry_date.isoformat() if cert.expiry_date else None
                    }
                    for cert in framework_certifications
                ]
            },
            "compliance_gaps": compliance_gaps,
            "data_governance": {
                "total_data_sources": len(self.data_inventory),
                "pii_data_sources": len([inv for inv in self.data_inventory.values() if inv.contains_pii]),
                "high_risk_data": len([inv for inv in self.data_inventory.values() if inv.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]]),
                "consent_required": len([inv for inv in self.data_inventory.values() if inv.consent_required]),
                "consent_obtained": len([inv for inv in self.data_inventory.values() if inv.consent_required and inv.consent_obtained])
            },
            "recommendations": recommendations,
            "next_actions": await self._generate_next_actions(framework, compliance_gaps)
        }
        
        return report
    
    async def monitor_compliance_status(self) -> Dict[str, Any]:
        """Monitor ongoing compliance status across all frameworks."""
        
        status_summary = {}
        
        for framework in ComplianceFramework:
            framework_certifications = [
                cert for cert in self.certifications.values()
                if cert.framework == framework
            ]
            
            if not framework_certifications:
                continue
            
            # Check for expiring certifications
            expiring_soon = [
                cert for cert in framework_certifications
                if cert.expiry_date and 
                cert.expiry_date <= datetime.utcnow() + timedelta(days=30)
            ]
            
            # Check for overdue reviews
            overdue_reviews = [
                cert for cert in framework_certifications
                if cert.next_review_date and 
                cert.next_review_date <= datetime.utcnow()
            ]
            
            status_summary[framework.value] = {
                "total_certifications": len(framework_certifications),
                "compliant": len([c for c in framework_certifications if c.status == ComplianceStatus.COMPLIANT]),
                "non_compliant": len([c for c in framework_certifications if c.status == ComplianceStatus.NON_COMPLIANT]),
                "expiring_soon": len(expiring_soon),
                "overdue_reviews": len(overdue_reviews),
                "average_compliance_score": sum(c.compliance_score for c in framework_certifications) / len(framework_certifications) if framework_certifications else 0
            }
        
        return status_summary
    
    async def _calculate_compliance_score(self, assessment_results: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score from assessment results."""
        
        if not assessment_results:
            return 0.0
        
        total_weight = 0
        weighted_score = 0
        
        for result in assessment_results:
            # Weight by risk level
            risk_weights = {
                GovernanceRisk.LOW: 1,
                GovernanceRisk.MEDIUM: 2,
                GovernanceRisk.HIGH: 3,
                GovernanceRisk.CRITICAL: 4
            }
            
            weight = risk_weights.get(result.get("risk_level", GovernanceRisk.MEDIUM), 2)
            score = 1.0 if result.get("compliant", False) else 0.0
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _determine_compliance_status(self,
                                         compliance_score: float,
                                         assessment_results: List[Dict[str, Any]]) -> ComplianceStatus:
        """Determine compliance status based on score and critical failures."""
        
        # Check for critical requirement failures
        critical_failures = [
            result for result in assessment_results
            if not result.get("compliant", False) and 
            result.get("risk_level") == GovernanceRisk.CRITICAL
        ]
        
        if critical_failures:
            return ComplianceStatus.NON_COMPLIANT
        
        # Score-based determination
        if compliance_score >= 0.95:
            return ComplianceStatus.COMPLIANT
        elif compliance_score >= 0.75:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    async def _calculate_overall_compliance_score(self,
                                                certifications: List[ComplianceCertification]) -> float:
        """Calculate overall compliance score across certifications."""
        
        if not certifications:
            return 0.0
        
        # Weight by recency and certification status
        total_weight = 0
        weighted_score = 0
        
        for cert in certifications:
            # Recent certifications have higher weight
            if cert.issued_date:
                days_old = (datetime.utcnow() - cert.issued_date).days
                recency_weight = max(0.1, 1.0 - (days_old / 365))
            else:
                recency_weight = 0.5
            
            # Status weight
            status_weights = {
                ComplianceStatus.COMPLIANT: 1.0,
                ComplianceStatus.PARTIALLY_COMPLIANT: 0.7,
                ComplianceStatus.NON_COMPLIANT: 0.0,
                ComplianceStatus.UNDER_REVIEW: 0.5,
                ComplianceStatus.EXPIRED: 0.0
            }
            
            status_weight = status_weights.get(cert.status, 0.5)
            
            weight = recency_weight
            score = cert.compliance_score * status_weight
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _identify_compliance_gaps(self,
                                      requirements: List[RegulatoryRequirement],
                                      certifications: List[ComplianceCertification]) -> List[Dict[str, Any]]:
        """Identify compliance gaps and areas for improvement."""
        
        gaps = []
        
        # Get all failed requirements across certifications
        failed_requirements = set()
        for cert in certifications:
            failed_requirements.update(cert.requirements_failed)
        
        # Analyze failed requirements
        for req_id in failed_requirements:
            requirement = next((r for r in requirements if r.requirement_id == req_id), None)
            if requirement:
                gaps.append({
                    "requirement_id": req_id,
                    "title": requirement.title,
                    "risk_level": requirement.risk_level.value,
                    "gap_type": "requirement_failure",
                    "description": f"Requirement '{requirement.title}' is not being met",
                    "remediation_priority": "high" if requirement.risk_level in [GovernanceRisk.HIGH, GovernanceRisk.CRITICAL] else "medium"
                })
        
        # Check for missing assessments
        assessed_requirements = set()
        for cert in certifications:
            assessed_requirements.update(cert.requirements_met + cert.requirements_failed)
        
        for requirement in requirements:
            if requirement.requirement_id not in assessed_requirements:
                gaps.append({
                    "requirement_id": requirement.requirement_id,
                    "title": requirement.title,
                    "risk_level": requirement.risk_level.value,
                    "gap_type": "missing_assessment",
                    "description": f"Requirement '{requirement.title}' has not been assessed",
                    "remediation_priority": "medium"
                })
        
        return gaps


class ComplianceAssessmentEngine:
    """Engine for performing automated compliance assessments."""
    
    async def assess_compliance(self,
                              model_id: str,
                              requirements: List[RegulatoryRequirement],
                              data_inventory: Dict[str, DataInventory]) -> List[Dict[str, Any]]:
        """Perform compliance assessment against requirements."""
        
        results = []
        
        for requirement in requirements:
            result = await self._assess_single_requirement(
                model_id, requirement, data_inventory
            )
            results.append(result)
        
        return results
    
    async def _assess_single_requirement(self,
                                       model_id: str,
                                       requirement: RegulatoryRequirement,
                                       data_inventory: Dict[str, DataInventory]) -> Dict[str, Any]:
        """Assess compliance with a single requirement."""
        
        result = {
            "requirement_id": requirement.requirement_id,
            "title": requirement.title,
            "framework": requirement.framework.value,
            "risk_level": requirement.risk_level,
            "compliant": False,
            "compliance_score": 0.0,
            "findings": [],
            "evidence": {}
        }
        
        # Perform specific assessments based on requirement type
        if "consent" in requirement.title.lower():
            compliance_check = await self._assess_consent_requirement(requirement, data_inventory)
        elif "data minimization" in requirement.title.lower():
            compliance_check = await self._assess_data_minimization(requirement, data_inventory)
        elif "encryption" in requirement.title.lower():
            compliance_check = await self._assess_encryption_requirement(requirement, data_inventory)
        elif "access control" in requirement.title.lower():
            compliance_check = await self._assess_access_controls(requirement, data_inventory)
        elif "audit" in requirement.title.lower():
            compliance_check = await self._assess_audit_requirements(requirement)
        else:
            # Generic assessment
            compliance_check = await self._generic_requirement_assessment(requirement, data_inventory)
        
        result.update(compliance_check)
        
        return result
    
    async def _assess_consent_requirement(self,
                                        requirement: RegulatoryRequirement,
                                        data_inventory: Dict[str, DataInventory]) -> Dict[str, Any]:
        """Assess consent-related requirements."""
        
        findings = []
        evidence = {}
        
        # Check if consent is required for any data sources
        consent_required_sources = [
            inv for inv in data_inventory.values()
            if inv.consent_required
        ]
        
        if not consent_required_sources:
            return {
                "compliant": True,
                "compliance_score": 1.0,
                "findings": ["No data sources require consent"],
                "evidence": {"consent_required_sources": 0}
            }
        
        # Check consent status
        consent_obtained_count = len([
            inv for inv in consent_required_sources
            if inv.consent_obtained
        ])
        
        compliance_score = consent_obtained_count / len(consent_required_sources) if consent_required_sources else 1.0
        
        if compliance_score < 1.0:
            findings.append(f"Consent missing for {len(consent_required_sources) - consent_obtained_count} data sources")
        
        evidence = {
            "consent_required_sources": len(consent_required_sources),
            "consent_obtained_sources": consent_obtained_count,
            "consent_compliance_rate": compliance_score
        }
        
        return {
            "compliant": compliance_score >= 0.95,
            "compliance_score": compliance_score,
            "findings": findings,
            "evidence": evidence
        }
    
    async def _assess_data_minimization(self,
                                      requirement: RegulatoryRequirement,
                                      data_inventory: Dict[str, DataInventory]) -> Dict[str, Any]:
        """Assess data minimization compliance."""
        
        findings = []
        evidence = {}
        
        # Check for documented processing purposes
        sources_with_purposes = [
            inv for inv in data_inventory.values()
            if inv.processing_purposes
        ]
        
        purpose_compliance = len(sources_with_purposes) / len(data_inventory) if data_inventory else 1.0
        
        # Check for high-sensitivity data without clear justification
        high_sensitivity_sources = [
            inv for inv in data_inventory.values()
            if inv.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]
            and not inv.processing_purposes
        ]
        
        if high_sensitivity_sources:
            findings.append(f"High-sensitivity data without documented purpose: {len(high_sensitivity_sources)} sources")
        
        compliance_score = purpose_compliance * (0.8 if high_sensitivity_sources else 1.0)
        
        evidence = {
            "total_data_sources": len(data_inventory),
            "sources_with_purposes": len(sources_with_purposes),
            "high_sensitivity_without_purpose": len(high_sensitivity_sources),
            "purpose_documentation_rate": purpose_compliance
        }
        
        return {
            "compliant": compliance_score >= 0.9,
            "compliance_score": compliance_score,
            "findings": findings,
            "evidence": evidence
        }
    
    async def _assess_encryption_requirement(self,
                                           requirement: RegulatoryRequirement,
                                           data_inventory: Dict[str, DataInventory]) -> Dict[str, Any]:
        """Assess encryption requirements."""
        
        findings = []
        evidence = {}
        
        # Check encryption status for sensitive data
        sensitive_data = [
            inv for inv in data_inventory.values()
            if inv.contains_pii or inv.contains_phi or 
            inv.data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED, DataClassification.TOP_SECRET]
        ]
        
        if not sensitive_data:
            return {
                "compliant": True,
                "compliance_score": 1.0,
                "findings": ["No sensitive data requiring encryption"],
                "evidence": {"sensitive_data_sources": 0}
            }
        
        # Check encryption at rest
        encrypted_at_rest = [
            inv for inv in sensitive_data
            if inv.encryption_at_rest
        ]
        
        # Check encryption in transit
        encrypted_in_transit = [
            inv for inv in sensitive_data
            if inv.encryption_in_transit
        ]
        
        at_rest_compliance = len(encrypted_at_rest) / len(sensitive_data)
        in_transit_compliance = len(encrypted_in_transit) / len(sensitive_data)
        
        if at_rest_compliance < 1.0:
            findings.append(f"Encryption at rest missing for {len(sensitive_data) - len(encrypted_at_rest)} sensitive data sources")
        
        if in_transit_compliance < 1.0:
            findings.append(f"Encryption in transit missing for {len(sensitive_data) - len(encrypted_in_transit)} sensitive data sources")
        
        compliance_score = (at_rest_compliance + in_transit_compliance) / 2
        
        evidence = {
            "sensitive_data_sources": len(sensitive_data),
            "encrypted_at_rest": len(encrypted_at_rest),
            "encrypted_in_transit": len(encrypted_in_transit),
            "at_rest_compliance_rate": at_rest_compliance,
            "in_transit_compliance_rate": in_transit_compliance
        }
        
        return {
            "compliant": compliance_score >= 0.95,
            "compliance_score": compliance_score,
            "findings": findings,
            "evidence": evidence
        }
    
    async def _generic_requirement_assessment(self,
                                            requirement: RegulatoryRequirement,
                                            data_inventory: Dict[str, DataInventory]) -> Dict[str, Any]:
        """Generic requirement assessment."""
        
        # Basic compliance check - in production would be more sophisticated
        compliance_score = 0.7  # Assume partial compliance
        
        return {
            "compliant": compliance_score >= 0.8,
            "compliance_score": compliance_score,
            "findings": ["Generic assessment - manual review required"],
            "evidence": {"assessment_type": "generic"}
        }


class PrivacyComplianceEngine:
    """Engine for privacy-specific compliance checks."""
    
    async def assess_privacy_compliance(self, data_inventory: DataInventory) -> Dict[str, Any]:
        """Assess privacy compliance for a data source."""
        
        compliance_issues = []
        
        # Check PII handling
        if data_inventory.contains_pii:
            if not data_inventory.legal_basis:
                compliance_issues.append("PII processing without legal basis")
            
            if data_inventory.consent_required and not data_inventory.consent_obtained:
                compliance_issues.append("Required consent not obtained")
            
            if not data_inventory.encryption_at_rest:
                compliance_issues.append("PII not encrypted at rest")
        
        # Check retention policies
        if data_inventory.contains_pii and not data_inventory.retention_period_days:
            compliance_issues.append("No retention period defined for PII")
        
        # Check cross-border transfers
        if data_inventory.cross_border_transfers and not data_inventory.data_residency_requirements:
            compliance_issues.append("Cross-border transfers without residency requirements")
        
        is_compliant = len(compliance_issues) == 0
        
        return {
            "compliant": is_compliant,
            "issues": compliance_issues,
            "risk_level": "high" if not is_compliant and data_inventory.contains_pii else "medium"
        }


class ComplianceReportGenerator:
    """Generates compliance reports and documentation."""
    
    async def generate_compliance_report(self,
                                       certification: ComplianceCertification,
                                       assessment_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive compliance assessment report."""
        
        report_content = {
            "executive_summary": {
                "certification_id": certification.certification_id,
                "framework": certification.framework.value,
                "assessment_date": datetime.utcnow().isoformat(),
                "overall_compliance_score": certification.compliance_score,
                "compliance_status": certification.status.value,
                "requirements_assessed": len(assessment_results),
                "requirements_met": len(certification.requirements_met),
                "requirements_failed": len(certification.requirements_failed)
            },
            "detailed_findings": assessment_results,
            "compliance_matrix": self._generate_compliance_matrix(assessment_results),
            "recommendations": self._generate_assessment_recommendations(assessment_results),
            "next_steps": self._generate_next_steps(certification, assessment_results)
        }
        
        # In production, would save to secure document storage
        report_path = f"/compliance/reports/{certification.certification_id}_assessment_report.json"
        
        # Would actually write file here
        # with open(report_path, 'w') as f:
        #     json.dump(report_content, f, indent=2)
        
        return report_path
    
    def _generate_compliance_matrix(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance matrix showing requirement status."""
        
        matrix = {
            "compliant": [],
            "non_compliant": [],
            "requires_review": []
        }
        
        for result in assessment_results:
            category = "compliant" if result.get("compliant", False) else "non_compliant"
            if result.get("compliance_score", 0) > 0.5 and not result.get("compliant", False):
                category = "requires_review"
            
            matrix[category].append({
                "requirement_id": result.get("requirement_id"),
                "title": result.get("title"),
                "compliance_score": result.get("compliance_score", 0)
            })
        
        return matrix


class EvidenceManager:
    """Manages compliance evidence and documentation."""
    
    def __init__(self):
        self.evidence_store: Dict[str, Any] = {}
        self.encryption_key = self._generate_encryption_key()
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive evidence."""
        # In production, would use proper key management
        return Fernet.generate_key()
    
    async def store_evidence(self,
                           evidence_id: str,
                           evidence_data: Dict[str, Any],
                           sensitive: bool = False) -> str:
        """Store compliance evidence securely."""
        
        if sensitive:
            # Encrypt sensitive evidence
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(json.dumps(evidence_data).encode())
            self.evidence_store[evidence_id] = {
                "encrypted": True,
                "data": encrypted_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            self.evidence_store[evidence_id] = {
                "encrypted": False,
                "data": evidence_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return evidence_id
    
    async def retrieve_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve compliance evidence."""
        
        if evidence_id not in self.evidence_store:
            return None
        
        evidence = self.evidence_store[evidence_id]
        
        if evidence["encrypted"]:
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(evidence["data"])
            return json.loads(decrypted_data.decode())
        else:
            return evidence["data"]


class RegulatoryComplianceMonitor:
    """Monitors ongoing regulatory compliance status."""
    
    def __init__(self):
        self.monitoring_rules: Dict[str, Any] = {}
    
    async def setup_compliance_monitoring(self,
                                        framework: ComplianceFramework,
                                        monitoring_rules: List[Dict[str, Any]]) -> None:
        """Setup automated compliance monitoring."""
        
        self.monitoring_rules[framework.value] = monitoring_rules
    
    async def check_compliance_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Check current compliance status for a framework."""
        
        # In production, would check actual system status
        return {
            "framework": framework.value,
            "status": "monitoring",
            "last_check": datetime.utcnow().isoformat(),
            "issues_detected": 0
        }