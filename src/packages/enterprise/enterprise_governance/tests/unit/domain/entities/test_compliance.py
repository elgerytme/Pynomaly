"""
Unit tests for Compliance domain entities.
"""

import pytest
from datetime import datetime, date, timedelta
from uuid import uuid4, UUID

from enterprise_governance.domain.entities.compliance import (
    ComplianceControl, ComplianceAssessment, ComplianceReport, DataPrivacyRecord,
    ComplianceFramework, ComplianceStatus, ControlStatus, EvidenceType
)


class TestComplianceControl:
    """Test cases for ComplianceControl entity."""
    
    def test_compliance_control_creation_basic(self):
        """Test basic compliance control creation."""
        control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC6.1",
            control_number="CC6.1",
            title="Logical and Physical Access Controls",
            description="The entity implements logical and physical access controls",
            category="Access Controls",
            domain="Security",
            risk_level="high"
        )
        
        assert isinstance(control.id, UUID)
        assert control.framework == ComplianceFramework.SOC2
        assert control.control_id == "CC6.1"
        assert control.control_number == "CC6.1"
        assert control.title == "Logical and Physical Access Controls"
        assert control.description == "The entity implements logical and physical access controls"
        assert control.category == "Access Controls"
        assert control.domain == "Security"
        assert control.status == ControlStatus.NOT_IMPLEMENTED
        assert control.risk_level == "high"
        assert control.priority == 3
        assert control.implementation_notes == ""
        assert control.remediation_plan == ""
        assert control.evidence_required == []
        assert control.evidence_collected == []
        assert control.automated_check is False
        assert control.tags == []
        
    def test_compliance_control_creation_comprehensive(self):
        """Test comprehensive compliance control creation."""
        tenant_id = uuid4()
        implementation_date = date.today() - timedelta(days=90)
        last_reviewed = date.today() - timedelta(days=30)
        next_review = date.today() + timedelta(days=335)
        evidence_required = [EvidenceType.DOCUMENT, EvidenceType.POLICY]
        evidence_collected = ["policy_doc.pdf", "implementation_guide.pdf"]
        tags = ["access_control", "security", "critical"]
        external_refs = ["NIST 800-53 AC-2", "ISO 27001 A.9.2.1"]
        
        control = ComplianceControl(
            framework=ComplianceFramework.GDPR,
            control_id="ART32",
            control_number="32",
            title="Security of processing",
            description="Taking into account technical and organisational measures",
            category="Data Security",
            subcategory="Technical Measures",
            domain="Privacy",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=implementation_date,
            last_reviewed_date=last_reviewed,
            next_review_date=next_review,
            risk_level="critical",
            priority=1,
            implementation_notes="Implemented with AES-256 encryption",
            remediation_plan="Regular security audits scheduled",
            responsible_party="Security Team",
            evidence_required=evidence_required,
            evidence_collected=evidence_collected,
            validation_method="Automated scanning + manual review",
            automated_check=True,
            automation_script="security_check.py",
            last_automated_check=datetime.utcnow() - timedelta(hours=24),
            tenant_id=tenant_id,
            tags=tags,
            external_references=external_refs
        )
        
        assert control.framework == ComplianceFramework.GDPR
        assert control.control_id == "ART32"
        assert control.subcategory == "Technical Measures"
        assert control.status == ControlStatus.IMPLEMENTED
        assert control.implementation_date == implementation_date
        assert control.last_reviewed_date == last_reviewed
        assert control.next_review_date == next_review
        assert control.risk_level == "critical"
        assert control.priority == 1
        assert control.implementation_notes == "Implemented with AES-256 encryption"
        assert control.responsible_party == "Security Team"
        assert control.evidence_required == evidence_required
        assert control.evidence_collected == evidence_collected
        assert control.validation_method == "Automated scanning + manual review"
        assert control.automated_check is True
        assert control.automation_script == "security_check.py"
        assert control.tenant_id == tenant_id
        assert control.tags == tags
        assert control.external_references == external_refs
        
    def test_is_compliant(self):
        """Test compliance status check."""
        implemented_control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            control_number="CC1.1",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium",
            status=ControlStatus.IMPLEMENTED
        )
        assert implemented_control.is_compliant() is True
        
        not_implemented_control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.2",
            control_number="CC1.2",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium",
            status=ControlStatus.NOT_IMPLEMENTED
        )
        assert not_implemented_control.is_compliant() is False
        
    def test_is_overdue_for_review(self):
        """Test overdue review check."""
        overdue_control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            control_number="CC1.1",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium",
            next_review_date=date.today() - timedelta(days=1)
        )
        assert overdue_control.is_overdue_for_review() is True
        
        not_overdue_control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.2",
            control_number="CC1.2",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium",
            next_review_date=date.today() + timedelta(days=30)
        )
        assert not_overdue_control.is_overdue_for_review() is False
        
        no_review_date_control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.3",
            control_number="CC1.3",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium"
        )
        assert no_review_date_control.is_overdue_for_review() is False
        
    def test_update_status(self):
        """Test control status update."""
        control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            control_number="CC1.1",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium"
        )
        
        original_updated_at = control.updated_at
        
        control.update_status(ControlStatus.IMPLEMENTED, "Control successfully implemented")
        
        assert control.status == ControlStatus.IMPLEMENTED
        assert control.implementation_date == date.today()
        assert control.updated_at > original_updated_at
        assert "Control successfully implemented" in control.implementation_notes
        
    def test_update_status_with_existing_notes(self):
        """Test status update with existing implementation notes."""
        control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            control_number="CC1.1",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium",
            implementation_notes="Initial notes"
        )
        
        control.update_status(ControlStatus.PARTIALLY_IMPLEMENTED, "Partial implementation")
        
        assert control.status == ControlStatus.PARTIALLY_IMPLEMENTED
        assert "Initial notes" in control.implementation_notes
        assert "Partial implementation" in control.implementation_notes
        
    def test_add_evidence(self):
        """Test adding evidence to control."""
        control = ComplianceControl(
            framework=ComplianceFramework.GDPR,
            control_id="ART32",
            control_number="32",
            title="Security of processing",
            description="Test description",
            category="Security",
            domain="Privacy",
            risk_level="high"
        )
        
        original_updated_at = control.updated_at
        
        control.add_evidence("security_policy.pdf", EvidenceType.POLICY)
        
        assert "security_policy.pdf" in control.evidence_collected
        assert control.updated_at > original_updated_at
        
        # Adding duplicate evidence should not duplicate
        control.add_evidence("security_policy.pdf", EvidenceType.POLICY)
        assert control.evidence_collected.count("security_policy.pdf") == 1
        
    def test_schedule_review(self):
        """Test scheduling next review."""
        control = ComplianceControl(
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            control_number="CC1.1",
            title="Test Control",
            description="Test control description",
            category="Test",
            domain="Test",
            risk_level="medium"
        )
        
        original_updated_at = control.updated_at
        
        control.schedule_review(6)  # 6 months ahead
        
        assert control.next_review_date is not None
        assert control.next_review_date > date.today()
        assert control.updated_at > original_updated_at


class TestComplianceAssessment:
    """Test cases for ComplianceAssessment entity."""
    
    def test_compliance_assessment_creation_basic(self):
        """Test basic compliance assessment creation."""
        tenant_id = uuid4()
        start_date = date.today()
        
        assessment = ComplianceAssessment(
            tenant_id=tenant_id,
            framework=ComplianceFramework.SOC2,
            assessment_name="SOC 2 Type II Assessment 2024",
            description="Annual SOC 2 Type II compliance assessment",
            scope="All production systems and processes",
            start_date=start_date,
            lead_assessor="Jane Smith, CISA"
        )
        
        assert isinstance(assessment.id, UUID)
        assert assessment.tenant_id == tenant_id
        assert assessment.framework == ComplianceFramework.SOC2
        assert assessment.assessment_name == "SOC 2 Type II Assessment 2024"
        assert assessment.description == "Annual SOC 2 Type II compliance assessment"
        assert assessment.scope == "All production systems and processes"
        assert assessment.start_date == start_date
        assert assessment.lead_assessor == "Jane Smith, CISA"
        assert assessment.end_date is None
        assert assessment.assessment_team == []
        assert assessment.overall_status == ComplianceStatus.UNDER_REVIEW
        assert assessment.compliance_percentage == 0.0
        assert assessment.certification_achieved is False
        
    def test_compliance_assessment_creation_comprehensive(self):
        """Test comprehensive compliance assessment creation."""
        tenant_id = uuid4()
        start_date = date.today() - timedelta(days=90)
        end_date = date.today() - timedelta(days=1)
        report_due = date.today() + timedelta(days=30)
        
        assessment = ComplianceAssessment(
            tenant_id=tenant_id,
            framework=ComplianceFramework.GDPR,
            assessment_name="GDPR Compliance Review 2024",
            description="Comprehensive GDPR compliance assessment",
            scope="All data processing activities",
            systems_in_scope=["CRM", "Analytics Platform", "User Database"],
            exclusions=["Marketing automation (third-party)"],
            start_date=start_date,
            end_date=end_date,
            report_due_date=report_due,
            lead_assessor="John Doe, CIPP/E",
            assessment_team=["Alice Johnson", "Bob Wilson"],
            external_auditor="External Audit Firm LLC",
            total_controls=50,
            controls_implemented=40,
            controls_partial=8,
            controls_not_implemented=2,
            overall_status=ComplianceStatus.PARTIAL_COMPLIANCE,
            compliance_percentage=80.0,
            risk_score=25.5,
            critical_findings=1,
            high_findings=3,
            medium_findings=5,
            low_findings=2,
            executive_summary="Assessment shows good compliance posture",
            recommendations=["Implement data retention policy", "Enhance access controls"],
            remediation_timeline="90 days for critical findings",
            certification_achieved=False,
            tags=["gdpr", "privacy", "annual"]
        )
        
        assert assessment.systems_in_scope == ["CRM", "Analytics Platform", "User Database"]
        assert assessment.exclusions == ["Marketing automation (third-party)"]
        assert assessment.end_date == end_date
        assert assessment.report_due_date == report_due
        assert assessment.assessment_team == ["Alice Johnson", "Bob Wilson"]
        assert assessment.external_auditor == "External Audit Firm LLC"
        assert assessment.total_controls == 50
        assert assessment.controls_implemented == 40
        assert assessment.controls_partial == 8
        assert assessment.controls_not_implemented == 2
        assert assessment.overall_status == ComplianceStatus.PARTIAL_COMPLIANCE
        assert assessment.compliance_percentage == 80.0
        assert assessment.risk_score == 25.5
        assert assessment.critical_findings == 1
        assert assessment.recommendations == ["Implement data retention policy", "Enhance access controls"]
        assert assessment.tags == ["gdpr", "privacy", "annual"]
        
    def test_update_control_counts(self):
        """Test updating control counts from control list."""
        assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Test Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today(),
            lead_assessor="Test Assessor"
        )
        
        # Create mock controls
        controls = [
            ComplianceControl(
                framework=ComplianceFramework.SOC2,
                control_id="CC1.1",
                control_number="CC1.1",
                title="Control 1",
                description="Description 1",
                category="Category 1",
                domain="Domain 1",
                risk_level="high",
                status=ControlStatus.IMPLEMENTED
            ),
            ComplianceControl(
                framework=ComplianceFramework.SOC2,
                control_id="CC1.2",
                control_number="CC1.2",
                title="Control 2",
                description="Description 2",
                category="Category 2",
                domain="Domain 2",
                risk_level="medium",
                status=ControlStatus.PARTIALLY_IMPLEMENTED
            ),
            ComplianceControl(
                framework=ComplianceFramework.SOC2,
                control_id="CC1.3",
                control_number="CC1.3",
                title="Control 3",
                description="Description 3",
                category="Category 3",
                domain="Domain 3",
                risk_level="low",
                status=ControlStatus.NOT_IMPLEMENTED
            )
        ]
        
        original_updated_at = assessment.updated_at
        
        assessment.update_control_counts(controls)
        
        assert assessment.total_controls == 3
        assert assessment.controls_implemented == 1
        assert assessment.controls_partial == 1
        assert assessment.controls_not_implemented == 1
        assert assessment.compliance_percentage == (1/3) * 100  # ~33.33%
        assert assessment.overall_status == ComplianceStatus.NON_COMPLIANT
        assert assessment.updated_at > original_updated_at
        
    def test_update_control_counts_high_compliance(self):
        """Test control counts update with high compliance."""
        assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Test Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today(),
            lead_assessor="Test Assessor"
        )
        
        # Create controls with high compliance
        controls = []
        for i in range(20):
            controls.append(
                ComplianceControl(
                    framework=ComplianceFramework.SOC2,
                    control_id=f"CC{i+1}.1",
                    control_number=f"CC{i+1}.1",
                    title=f"Control {i+1}",
                    description=f"Description {i+1}",
                    category="Category",
                    domain="Domain",
                    risk_level="medium",
                    status=ControlStatus.IMPLEMENTED if i < 19 else ControlStatus.NOT_IMPLEMENTED
                )
            )
        
        assessment.update_control_counts(controls)
        
        assert assessment.total_controls == 20
        assert assessment.controls_implemented == 19
        assert assessment.controls_not_implemented == 1
        assert assessment.compliance_percentage == 95.0
        assert assessment.overall_status == ComplianceStatus.COMPLIANT
        
    def test_add_finding(self):
        """Test adding findings to assessment."""
        assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Test Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today(),
            lead_assessor="Test Assessor"
        )
        
        original_updated_at = assessment.updated_at
        
        assessment.add_finding("critical")
        assessment.add_finding("high")
        assessment.add_finding("medium")
        assessment.add_finding("low")
        
        assert assessment.critical_findings == 1
        assert assessment.high_findings == 1
        assert assessment.medium_findings == 1
        assert assessment.low_findings == 1
        assert assessment.updated_at > original_updated_at
        
    def test_complete_assessment(self):
        """Test completing assessment."""
        assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Test Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today() - timedelta(days=30),
            lead_assessor="Test Assessor",
            overall_status=ComplianceStatus.UNDER_REVIEW,
            compliance_percentage=85.0
        )
        
        assessment.complete_assessment()
        
        assert assessment.end_date == date.today()
        assert assessment.completed_at is not None
        assert assessment.overall_status == ComplianceStatus.PARTIAL_COMPLIANCE  # 85% -> partial
        
    def test_is_overdue(self):
        """Test overdue assessment check."""
        overdue_assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Overdue Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today() - timedelta(days=60),
            lead_assessor="Test Assessor",
            report_due_date=date.today() - timedelta(days=1)
        )
        assert overdue_assessment.is_overdue() is True
        
        not_overdue_assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="On Time Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today() - timedelta(days=30),
            lead_assessor="Test Assessor",
            report_due_date=date.today() + timedelta(days=30)
        )
        assert not_overdue_assessment.is_overdue() is False
        
        completed_assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.SOC2,
            assessment_name="Completed Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today() - timedelta(days=60),
            lead_assessor="Test Assessor",
            report_due_date=date.today() - timedelta(days=1),
            completed_at=datetime.utcnow()
        )
        assert completed_assessment.is_overdue() is False
        
    def test_get_compliance_summary(self):
        """Test getting compliance summary."""
        assessment = ComplianceAssessment(
            tenant_id=uuid4(),
            framework=ComplianceFramework.GDPR,
            assessment_name="GDPR Assessment",
            description="Test description",
            scope="Test scope",
            start_date=date.today(),
            lead_assessor="Test Assessor",
            total_controls=100,
            controls_implemented=85,
            controls_partial=10,
            controls_not_implemented=5,
            overall_status=ComplianceStatus.PARTIAL_COMPLIANCE,
            compliance_percentage=85.0,
            critical_findings=2,
            high_findings=5,
            medium_findings=8,
            low_findings=3,
            certification_achieved=False
        )
        
        summary = assessment.get_compliance_summary()
        
        assert summary["framework"] == ComplianceFramework.GDPR
        assert summary["overall_status"] == ComplianceStatus.PARTIAL_COMPLIANCE
        assert summary["compliance_percentage"] == 85.0
        assert summary["total_controls"] == 100
        assert summary["implemented"] == 85
        assert summary["partial"] == 10
        assert summary["not_implemented"] == 5
        assert summary["critical_findings"] == 2
        assert summary["certification_achieved"] is False
        assert "is_overdue" in summary


class TestComplianceReport:
    """Test cases for ComplianceReport entity."""
    
    def test_compliance_report_creation(self):
        """Test compliance report creation."""
        tenant_id = uuid4()
        assessment_id = uuid4()
        period_start = date.today() - timedelta(days=365)
        period_end = date.today()
        
        report = ComplianceReport(
            tenant_id=tenant_id,
            assessment_id=assessment_id,
            report_type="SOC 2 Type II",
            title="SOC 2 Type II Examination Report",
            framework=ComplianceFramework.SOC2,
            report_period_start=period_start,
            report_period_end=period_end,
            generated_by="Audit System",
            executive_summary="The examination found the controls were effective",
            methodology="Testing and inquiry procedures",
            scope_and_limitations="Scope limited to production systems",
            compliance_score=92.5,
            risk_rating="Low",
            control_effectiveness="Effective"
        )
        
        assert isinstance(report.id, UUID)
        assert report.tenant_id == tenant_id
        assert report.assessment_id == assessment_id
        assert report.report_type == "SOC 2 Type II"
        assert report.title == "SOC 2 Type II Examination Report"
        assert report.framework == ComplianceFramework.SOC2
        assert report.report_period_start == period_start
        assert report.report_period_end == period_end
        assert report.generated_by == "Audit System"
        assert report.compliance_score == 92.5
        assert report.risk_rating == "Low"
        assert report.control_effectiveness == "Effective"
        assert report.status == "draft"
        assert report.version == "1.0"
        assert report.findings == []
        assert report.recommendations == []
        
    def test_add_finding(self):
        """Test adding finding to report."""
        report = ComplianceReport(
            tenant_id=uuid4(),
            assessment_id=uuid4(),
            report_type="GDPR Assessment",
            title="GDPR Compliance Report",
            framework=ComplianceFramework.GDPR,
            report_period_start=date.today() - timedelta(days=90),
            report_period_end=date.today(),
            generated_by="Compliance Team",
            executive_summary="Assessment summary",
            methodology="Assessment methodology",
            scope_and_limitations="Scope and limitations",
            compliance_score=85.0,
            risk_rating="Medium",
            control_effectiveness="Mostly Effective"
        )
        
        original_updated_at = report.updated_at
        
        report.add_finding(
            control_id="ART32",
            title="Encryption Not Implemented",
            description="Data at rest is not encrypted",
            risk_level="high",
            recommendation="Implement AES-256 encryption"
        )
        
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding["control_id"] == "ART32"
        assert finding["title"] == "Encryption Not Implemented"
        assert finding["description"] == "Data at rest is not encrypted"
        assert finding["risk_level"] == "high"
        assert finding["recommendation"] == "Implement AES-256 encryption"
        assert "id" in finding
        assert "identified_at" in finding
        assert report.updated_at > original_updated_at
        
    def test_add_recommendation(self):
        """Test adding recommendation to report."""
        report = ComplianceReport(
            tenant_id=uuid4(),
            assessment_id=uuid4(),
            report_type="ISO 27001 Assessment",
            title="ISO 27001 Compliance Report",
            framework=ComplianceFramework.ISO27001,
            report_period_start=date.today() - timedelta(days=90),
            report_period_end=date.today(),
            generated_by="Security Team",
            executive_summary="Assessment summary",
            methodology="Assessment methodology",
            scope_and_limitations="Scope and limitations",
            compliance_score=78.0,
            risk_rating="Medium",
            control_effectiveness="Partially Effective"
        )
        
        original_updated_at = report.updated_at
        
        report.add_recommendation(
            priority="high",
            title="Implement MFA",
            description="Multi-factor authentication should be implemented",
            timeline="60 days",
            responsible_party="IT Security Team"
        )
        
        assert len(report.recommendations) == 1
        recommendation = report.recommendations[0]
        assert recommendation["priority"] == "high"
        assert recommendation["title"] == "Implement MFA"
        assert recommendation["description"] == "Multi-factor authentication should be implemented"
        assert recommendation["timeline"] == "60 days"
        assert recommendation["responsible_party"] == "IT Security Team"
        assert "id" in recommendation
        assert "created_at" in recommendation
        assert report.updated_at > original_updated_at
        
    def test_finalize_report(self):
        """Test finalizing report."""
        report = ComplianceReport(
            tenant_id=uuid4(),
            assessment_id=uuid4(),
            report_type="HIPAA Assessment",
            title="HIPAA Compliance Report",
            framework=ComplianceFramework.HIPAA,
            report_period_start=date.today() - timedelta(days=90),
            report_period_end=date.today(),
            generated_by="Compliance Officer",
            executive_summary="Assessment summary",
            methodology="Assessment methodology",
            scope_and_limitations="Scope and limitations",
            compliance_score=95.0,
            risk_rating="Low",
            control_effectiveness="Effective"
        )
        
        approved_by = "Chief Compliance Officer"
        original_updated_at = report.updated_at
        
        report.finalize_report(approved_by)
        
        assert report.status == "approved"
        assert report.approved_by == approved_by
        assert report.published_at is not None
        assert report.updated_at > original_updated_at
        
    def test_get_findings_by_risk(self):
        """Test filtering findings by risk level."""
        report = ComplianceReport(
            tenant_id=uuid4(),
            assessment_id=uuid4(),
            report_type="Test Report",
            title="Test Compliance Report",
            framework=ComplianceFramework.SOC2,
            report_period_start=date.today() - timedelta(days=90),
            report_period_end=date.today(),
            generated_by="Test User",
            executive_summary="Test summary",
            methodology="Test methodology",
            scope_and_limitations="Test scope",
            compliance_score=85.0,
            risk_rating="Medium",
            control_effectiveness="Effective"
        )
        
        # Add findings with different risk levels
        report.add_finding("C1", "High Risk Finding", "Description", "high", "Recommendation")
        report.add_finding("C2", "Medium Risk Finding", "Description", "medium", "Recommendation")
        report.add_finding("C3", "Another High Risk Finding", "Description", "high", "Recommendation")
        
        high_risk_findings = report.get_findings_by_risk("high")
        assert len(high_risk_findings) == 2
        
        medium_risk_findings = report.get_findings_by_risk("medium")
        assert len(medium_risk_findings) == 1
        
        low_risk_findings = report.get_findings_by_risk("low")
        assert len(low_risk_findings) == 0
        
    def test_get_high_priority_recommendations(self):
        """Test getting high priority recommendations."""
        report = ComplianceReport(
            tenant_id=uuid4(),
            assessment_id=uuid4(),
            report_type="Test Report",
            title="Test Compliance Report",
            framework=ComplianceFramework.SOC2,
            report_period_start=date.today() - timedelta(days=90),
            report_period_end=date.today(),
            generated_by="Test User",
            executive_summary="Test summary",
            methodology="Test methodology",
            scope_and_limitations="Test scope",
            compliance_score=85.0,
            risk_rating="Medium",
            control_effectiveness="Effective"
        )
        
        # Add recommendations with different priorities
        report.add_recommendation("critical", "Critical Rec", "Description", "30 days", "Team A")
        report.add_recommendation("high", "High Rec", "Description", "60 days", "Team B")
        report.add_recommendation("medium", "Medium Rec", "Description", "90 days", "Team C")
        report.add_recommendation("low", "Low Rec", "Description", "120 days", "Team D")
        
        high_priority_recs = report.get_high_priority_recommendations()
        assert len(high_priority_recs) == 2  # critical and high
        
        priorities = [rec["priority"] for rec in high_priority_recs]
        assert "critical" in priorities
        assert "high" in priorities
        assert "medium" not in priorities


class TestDataPrivacyRecord:
    """Test cases for DataPrivacyRecord entity."""
    
    def test_data_privacy_record_creation(self):
        """Test data privacy record creation."""
        tenant_id = uuid4()
        retention_start = date.today()
        
        record = DataPrivacyRecord(
            tenant_id=tenant_id,
            data_subject_id="user123",
            record_type="customer_data",
            processing_purpose="Service delivery",
            data_categories=["personal_identifiers", "contact_information"],
            legal_basis="Contract performance",
            retention_period="7 years",
            retention_start_date=retention_start
        )
        
        assert isinstance(record.id, UUID)
        assert record.tenant_id == tenant_id
        assert record.data_subject_id == "user123"
        assert record.record_type == "customer_data"
        assert record.processing_purpose == "Service delivery"
        assert record.data_categories == ["personal_identifiers", "contact_information"]
        assert record.legal_basis == "Contract performance"
        assert record.consent_given is False
        assert record.retention_period == "7 years"
        assert record.retention_start_date == retention_start
        assert record.data_shared_with == []
        assert record.transfer_countries == []
        assert record.access_requests == []
        assert record.breach_incidents == []
        
    def test_data_privacy_record_with_consent(self):
        """Test data privacy record creation with consent."""
        tenant_id = uuid4()
        consent_date = datetime.utcnow() - timedelta(days=30)
        retention_start = date.today() - timedelta(days=30)
        scheduled_deletion = date.today() + timedelta(days=2555)  # 7 years from retention start
        
        record = DataPrivacyRecord(
            tenant_id=tenant_id,
            data_subject_id="user456",
            record_type="marketing_data",
            processing_purpose="Marketing communications",
            data_categories=["contact_information", "preferences"],
            legal_basis="Consent",
            consent_given=True,
            consent_date=consent_date,
            consent_method="Online form",
            retention_period="Until consent withdrawn",
            retention_start_date=retention_start,
            scheduled_deletion_date=scheduled_deletion,
            data_shared_with=["Marketing Partner Inc"],
            transfer_countries=["United States"],
            safeguards_in_place=["Standard Contractual Clauses"]
        )
        
        assert record.consent_given is True
        assert record.consent_date == consent_date
        assert record.consent_method == "Online form"
        assert record.scheduled_deletion_date == scheduled_deletion
        assert record.data_shared_with == ["Marketing Partner Inc"]
        assert record.transfer_countries == ["United States"]
        assert record.safeguards_in_place == ["Standard Contractual Clauses"]
        
    def test_withdraw_consent(self):
        """Test consent withdrawal."""
        record = DataPrivacyRecord(
            tenant_id=uuid4(),
            data_subject_id="user789",
            record_type="newsletter",
            processing_purpose="Newsletter delivery",
            data_categories=["email"],
            legal_basis="Consent",
            consent_given=True,
            consent_date=datetime.utcnow() - timedelta(days=60),
            retention_period="Until consent withdrawn",
            retention_start_date=date.today() - timedelta(days=60)
        )
        
        original_updated_at = record.updated_at
        
        record.withdraw_consent()
        
        assert record.consent_given is False
        assert record.consent_withdrawn_date is not None
        assert record.updated_at > original_updated_at
        
    def test_is_retention_expired(self):
        """Test retention expiry check."""
        expired_record = DataPrivacyRecord(
            tenant_id=uuid4(),
            data_subject_id="user_expired",
            record_type="temp_data",
            processing_purpose="Temporary processing",
            data_categories=["temp_data"],
            legal_basis="Legitimate interest",
            retention_period="30 days",
            retention_start_date=date.today() - timedelta(days=31),
            scheduled_deletion_date=date.today() - timedelta(days=1)
        )
        assert expired_record.is_retention_expired() is True
        
        not_expired_record = DataPrivacyRecord(
            tenant_id=uuid4(),
            data_subject_id="user_active",
            record_type="active_data",
            processing_purpose="Active processing",
            data_categories=["active_data"],
            legal_basis="Contract",
            retention_period="1 year",
            retention_start_date=date.today() - timedelta(days=180),
            scheduled_deletion_date=date.today() + timedelta(days=185)
        )
        assert not_expired_record.is_retention_expired() is False
        
        no_deletion_date_record = DataPrivacyRecord(
            tenant_id=uuid4(),
            data_subject_id="user_no_date",
            record_type="permanent_data",
            processing_purpose="Permanent processing",
            data_categories=["permanent_data"],
            legal_basis="Legal obligation",
            retention_period="Indefinite",
            retention_start_date=date.today()
        )
        assert no_deletion_date_record.is_retention_expired() is False
        
    def test_add_rights_request(self):
        """Test adding data subject rights requests."""
        record = DataPrivacyRecord(
            tenant_id=uuid4(),
            data_subject_id="user_rights",
            record_type="customer_data",
            processing_purpose="Service delivery",
            data_categories=["personal_data"],
            legal_basis="Contract",
            retention_period="5 years",
            retention_start_date=date.today()
        )
        
        original_updated_at = record.updated_at
        
        # Add different types of requests
        record.add_rights_request("access")
        record.add_rights_request("rectification")
        record.add_rights_request("erasure")
        record.add_rights_request("portability")
        
        assert len(record.access_requests) == 1
        assert len(record.rectification_requests) == 1
        assert len(record.erasure_requests) == 1
        assert len(record.portability_requests) == 1
        assert record.updated_at > original_updated_at
        
        # Add multiple access requests
        record.add_rights_request("access")
        assert len(record.access_requests) == 2
        
        # Invalid request type should not add anything
        original_access_count = len(record.access_requests)
        record.add_rights_request("invalid_type")
        assert len(record.access_requests) == original_access_count