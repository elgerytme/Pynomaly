"""Tests for governance framework service."""

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from pynomaly.application.services.governance_framework_service import (
    GovernanceFrameworkService, GovernanceAction, PolicyType, RiskLevel,
    ApprovalStatus, AuditTrailEntry, GovernancePolicy, RiskAssessment,
    ChangeRequest, ComplianceMetric
)
from pynomaly.domain.entities.security_compliance import ComplianceFramework, AuditLevel


class TestGovernanceFrameworkService:
    """Test cases for governance framework service."""
    
    @pytest.fixture
    def service(self, tmp_path):
        """Create governance service instance."""
        return GovernanceFrameworkService(
            storage_path=tmp_path,
            retention_years=7
        )
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service, tmp_path):
        """Test service initialization."""
        assert service.storage_path == tmp_path
        assert service.storage_path.exists()
        assert service.retention_years == 7
        assert isinstance(service.audit_trail, list)
        assert isinstance(service.policies, dict)
        assert isinstance(service.risk_assessments, dict)
        assert isinstance(service.change_requests, dict)
        assert isinstance(service.compliance_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_log_audit_event(self, service):
        """Test audit event logging."""
        entry_id = await service.log_audit_event(
            user_id="test_user",
            action=GovernanceAction.CREATE,
            resource_type="dataset",
            resource_id="dataset_123",
            details="Created new dataset for analysis",
            context={"project": "anomaly_detection"},
            risk_level=RiskLevel.LOW,
            compliance_frameworks=[ComplianceFramework.SOC2]
        )
        
        assert entry_id is not None
        assert len(service.audit_trail) == 1
        
        entry = service.audit_trail[0]
        assert entry.entry_id == entry_id
        assert entry.user_id == "test_user"
        assert entry.action == GovernanceAction.CREATE
        assert entry.resource_type == "dataset"
        assert entry.resource_id == "dataset_123"
        assert entry.details == "Created new dataset for analysis"
        assert entry.context["project"] == "anomaly_detection"
        assert entry.risk_assessment == RiskLevel.LOW
        assert ComplianceFramework.SOC2 in entry.compliance_frameworks
    
    @pytest.mark.asyncio
    async def test_create_policy(self, service):
        """Test governance policy creation."""
        policy_content = {
            "description": "Data retention policy for ML datasets",
            "retention_period_days": 365,
            "approval_required": True
        }
        
        policy_id = await service.create_policy(
            policy_name="Data Retention Policy",
            policy_type=PolicyType.DATA_GOVERNANCE,
            description="Policy for managing data retention periods",
            policy_content=policy_content,
            created_by="policy_admin",
            applicable_roles=["data_scientist", "ml_engineer"],
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2]
        )
        
        assert policy_id in service.policies
        policy = service.policies[policy_id]
        
        assert policy.policy_name == "Data Retention Policy"
        assert policy.policy_type == PolicyType.DATA_GOVERNANCE
        assert policy.created_by == "policy_admin"
        assert "data_scientist" in policy.applicable_roles
        assert "ml_engineer" in policy.applicable_roles
        assert ComplianceFramework.GDPR in policy.compliance_frameworks
        assert ComplianceFramework.SOC2 in policy.compliance_frameworks
        assert policy.policy_content["retention_period_days"] == 365
        assert policy.is_active()
        
        # Check audit trail
        assert len(service.audit_trail) == 1
        audit_entry = service.audit_trail[0]
        assert audit_entry.action == GovernanceAction.CREATE
        assert audit_entry.resource_type == "policy"
        assert audit_entry.resource_id == policy_id
    
    @pytest.mark.asyncio
    async def test_conduct_risk_assessment(self, service):
        """Test risk assessment functionality."""
        assessment_id = await service.conduct_risk_assessment(
            assessor="risk_analyst",
            risk_category="Data Security",
            risk_description="Potential unauthorized access to ML training data",
            likelihood=RiskLevel.MEDIUM,
            impact=RiskLevel.HIGH,
            affected_assets=["training_dataset", "ml_models", "user_data"],
            threat_sources=["external_attackers", "insider_threats"],
            existing_controls=["access_controls", "encryption", "monitoring"]
        )
        
        assert assessment_id in service.risk_assessments
        assessment = service.risk_assessments[assessment_id]
        
        assert assessment.assessor == "risk_analyst"
        assert assessment.risk_category == "Data Security"
        assert assessment.likelihood == RiskLevel.MEDIUM
        assert assessment.impact == RiskLevel.HIGH
        assert assessment.overall_risk == RiskLevel.HIGH  # Medium * High = High
        assert "training_dataset" in assessment.affected_assets
        assert "external_attackers" in assessment.threat_sources
        assert "access_controls" in assessment.existing_controls
        
        # Test risk score calculation
        risk_score = assessment.calculate_risk_score()
        assert risk_score == 12  # Medium (3) * High (4) = 12
        
        # Check audit trail
        assert len(service.audit_trail) == 1
        audit_entry = service.audit_trail[0]
        assert audit_entry.action == GovernanceAction.CREATE
        assert audit_entry.resource_type == "risk_assessment"
        assert audit_entry.risk_assessment == RiskLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_submit_change_request(self, service):
        """Test change management request submission."""
        request_id = await service.submit_change_request(
            title="Update ML Model Training Pipeline",
            description="Implement new feature engineering pipeline for improved model accuracy",
            change_type="system",
            requested_by="ml_engineer",
            approvers=["team_lead", "security_officer", "data_owner"],
            urgency="normal",
            impact_analysis="Medium impact - affects training performance but not production models",
            rollback_plan="Revert to previous pipeline configuration",
            affected_systems=["training_pipeline", "feature_store", "model_registry"]
        )
        
        assert request_id in service.change_requests
        change_request = service.change_requests[request_id]
        
        assert change_request.title == "Update ML Model Training Pipeline"
        assert change_request.change_type == "system"
        assert change_request.requested_by == "ml_engineer"
        assert change_request.approval_status == ApprovalStatus.PENDING
        assert "team_lead" in change_request.approvers
        assert "security_officer" in change_request.approvers
        assert "data_owner" in change_request.approvers
        assert "training_pipeline" in change_request.affected_systems
        
        # Check audit trail
        assert len(service.audit_trail) == 1
        audit_entry = service.audit_trail[0]
        assert audit_entry.action == GovernanceAction.CREATE
        assert audit_entry.resource_type == "change_request"
    
    @pytest.mark.asyncio
    async def test_approve_change_request(self, service):
        """Test change request approval workflow."""
        # First submit a change request
        request_id = await service.submit_change_request(
            title="Test Change",
            description="Test change for approval workflow",
            change_type="configuration",
            requested_by="requester",
            approvers=["approver1", "approver2"],
            urgency="normal"
        )
        
        # First approval
        is_fully_approved = await service.approve_change_request(
            request_id=request_id,
            approver="approver1",
            decision="approved",
            comments="Looks good from technical perspective"
        )
        
        assert not is_fully_approved  # Still needs second approval
        change_request = service.change_requests[request_id]
        assert change_request.approval_status == ApprovalStatus.PENDING
        assert len(change_request.approval_history) == 1
        assert change_request.approval_history[0]["approver"] == "approver1"
        assert change_request.approval_history[0]["decision"] == "approved"
        
        # Second approval - should complete the approval
        is_fully_approved = await service.approve_change_request(
            request_id=request_id,
            approver="approver2",
            decision="approved",
            comments="Security review complete"
        )
        
        assert is_fully_approved
        change_request = service.change_requests[request_id]
        assert change_request.approval_status == ApprovalStatus.APPROVED
        assert len(change_request.approval_history) == 2
        
        # Check audit trail (should have 3 entries: create + 2 approvals)
        assert len(service.audit_trail) == 3
        
        # Test rejection
        request_id2 = await service.submit_change_request(
            title="Test Rejection",
            description="Test change for rejection",
            change_type="policy",
            requested_by="requester",
            approvers=["approver1"],
            urgency="low"
        )
        
        is_approved = await service.approve_change_request(
            request_id=request_id2,
            approver="approver1",
            decision="rejected",
            comments="Security concerns"
        )
        
        assert not is_approved
        change_request2 = service.change_requests[request_id2]
        assert change_request2.approval_status == ApprovalStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_track_compliance_metric(self, service):
        """Test compliance metric tracking."""
        metric_id = await service.track_compliance_metric(
            metric_name="Data Encryption Coverage",
            framework=ComplianceFramework.SOC2,
            control_id="CC6.1",
            current_value=95.5,
            target_value=100.0,
            responsible_party="security_team",
            evidence=["encryption_audit_report.pdf", "key_management_review.docx"]
        )
        
        assert metric_id in service.compliance_metrics
        metric = service.compliance_metrics[metric_id]
        
        assert metric.metric_name == "Data Encryption Coverage"
        assert metric.framework == ComplianceFramework.SOC2
        assert metric.control_id == "CC6.1"
        assert metric.current_value == 95.5
        assert metric.target_value == 100.0
        assert metric.responsible_party == "security_team"
        assert "encryption_audit_report.pdf" in metric.evidence_collected
        
        # Test compliance status
        assert metric.get_compliance_status() == "compliant"  # 95.5 >= target (100) - threshold
        
        # Test non-compliant metric
        metric_id2 = await service.track_compliance_metric(
            metric_name="Access Review Completion",
            framework=ComplianceFramework.SOC2,
            control_id="CC6.2",
            current_value=65.0,
            target_value=100.0,
            responsible_party="access_team"
        )
        
        metric2 = service.compliance_metrics[metric_id2]
        assert metric2.get_compliance_status() == "non_compliant"  # Below critical threshold
        
        # Should have created a compliance violation
        assert len(service.compliance_violations) == 1
        violation = service.compliance_violations[0]
        assert violation["metric_id"] == metric_id2
        assert violation["status"] == "non_compliant"
        
        # Check audit trail
        assert len(service.audit_trail) == 2  # Two metric tracking events
    
    @pytest.mark.asyncio
    async def test_generate_audit_report(self, service):
        """Test audit report generation."""
        # Create some audit events
        await service.log_audit_event(
            user_id="user1", action=GovernanceAction.CREATE, resource_type="dataset",
            resource_id="ds1", details="Created dataset", risk_level=RiskLevel.LOW
        )
        await service.log_audit_event(
            user_id="user2", action=GovernanceAction.UPDATE, resource_type="model",
            resource_id="m1", details="Updated model", risk_level=RiskLevel.MEDIUM
        )
        await service.log_audit_event(
            user_id="user1", action=GovernanceAction.DELETE, resource_type="dataset",
            resource_id="ds2", details="Deleted dataset", risk_level=RiskLevel.HIGH
        )
        
        # Generate report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        report = await service.generate_audit_report(
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify report structure
        assert "report_id" in report
        assert "generation_date" in report
        assert "period" in report
        assert "summary" in report
        assert "action_distribution" in report
        assert "risk_distribution" in report
        assert "audit_entries" in report
        
        # Verify summary
        summary = report["summary"]
        assert summary["total_events"] == 3
        assert summary["unique_users"] == 2
        assert summary["unique_resources"] == 3
        
        # Verify action distribution
        action_dist = report["action_distribution"]
        assert action_dist["create"] == 1
        assert action_dist["update"] == 1
        assert action_dist["delete"] == 1
        
        # Verify risk distribution
        risk_dist = report["risk_distribution"]
        assert risk_dist["low"] == 1
        assert risk_dist["medium"] == 1
        assert risk_dist["high"] == 1
        
        # Test filtered report
        filtered_report = await service.generate_audit_report(
            start_date=start_date,
            end_date=end_date,
            resource_types=["dataset"],
            users=["user1"]
        )
        
        # Should only include user1's dataset events
        assert filtered_report["summary"]["total_events"] == 2
        assert filtered_report["summary"]["unique_users"] == 1
    
    @pytest.mark.asyncio
    async def test_get_governance_dashboard(self, service):
        """Test governance dashboard generation."""
        # Create some test data
        await service.create_policy(
            policy_name="Test Policy",
            policy_type=PolicyType.SECURITY,
            description="Test policy",
            policy_content={},
            created_by="admin"
        )
        
        await service.conduct_risk_assessment(
            assessor="analyst",
            risk_category="Test Risk",
            risk_description="Test risk description",
            likelihood=RiskLevel.HIGH,
            impact=RiskLevel.CRITICAL,
            affected_assets=["asset1"]
        )
        
        await service.submit_change_request(
            title="Test Change",
            description="Test change",
            change_type="system",
            requested_by="user",
            approvers=["approver"]
        )
        
        await service.track_compliance_metric(
            metric_name="Test Metric",
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            current_value=90.0,
            target_value=95.0,
            responsible_party="team"
        )
        
        # Generate dashboard
        dashboard = await service.get_governance_dashboard()
        
        # Verify dashboard structure
        assert "dashboard_id" in dashboard
        assert "generated_at" in dashboard
        assert "audit_activity" in dashboard
        assert "policy_management" in dashboard
        assert "risk_management" in dashboard
        assert "change_management" in dashboard
        assert "compliance_monitoring" in dashboard
        
        # Verify data
        assert dashboard["policy_management"]["active_policies"] == 1
        assert dashboard["risk_management"]["total_assessments"] == 1
        assert dashboard["risk_management"]["high_risk_assessments"] == 1  # HIGH + CRITICAL = high risk
        assert dashboard["change_management"]["total_requests"] == 1
        assert dashboard["change_management"]["pending_changes"] == 1
        assert dashboard["compliance_monitoring"]["total_metrics"] == 1
    
    @pytest.mark.asyncio
    async def test_audit_indices_update(self, service):
        """Test audit trail indexing for fast searching."""
        await service.log_audit_event(
            user_id="test_user",
            action=GovernanceAction.ACCESS,
            resource_type="dataset",
            resource_id="ds_123",
            details="Accessed dataset"
        )
        
        # Check that indices were updated
        assert "test_user" in service.audit_indices
        assert "dataset:ds_123" in service.audit_indices
        
        entry_id = service.audit_trail[0].entry_id
        assert entry_id in service.audit_indices["test_user"]
        assert entry_id in service.audit_indices["dataset:ds_123"]
    
    @pytest.mark.asyncio
    async def test_audit_entry_integrity(self, service):
        """Test audit entry hash generation for integrity."""
        await service.log_audit_event(
            user_id="test_user",
            action=GovernanceAction.CREATE,
            resource_type="model",
            resource_id="model_456",
            details="Created ML model"
        )
        
        entry = service.audit_trail[0]
        hash1 = entry.get_audit_hash()
        
        # Hash should be consistent
        hash2 = entry.get_audit_hash()
        assert hash1 == hash2
        
        # Hash should change if content changes
        entry.details = "Modified details"
        hash3 = entry.get_audit_hash()
        assert hash1 != hash3
    
    def test_audit_trail_entry_serialization(self):
        """Test audit trail entry serialization."""
        entry = AuditTrailEntry(
            user_id="test_user",
            action=GovernanceAction.UPDATE,
            resource_type="policy",
            resource_id="policy_123",
            details="Updated policy content",
            context={"version": "2.0"},
            risk_assessment=RiskLevel.MEDIUM,
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2]
        )
        
        serialized = entry.to_dict()
        
        assert serialized["user_id"] == "test_user"
        assert serialized["action"] == "update"
        assert serialized["resource_type"] == "policy"
        assert serialized["resource_id"] == "policy_123"
        assert serialized["details"] == "Updated policy content"
        assert serialized["context"]["version"] == "2.0"
        assert serialized["risk_assessment"] == "medium"
        assert "gdpr" in serialized["compliance_frameworks"]
        assert "soc2" in serialized["compliance_frameworks"]
        assert "audit_hash" in serialized
        assert "timestamp" in serialized
    
    def test_governance_policy_validation(self):
        """Test governance policy validation."""
        # Test valid policy
        policy = GovernancePolicy(
            policy_name="Test Policy",
            policy_type=PolicyType.DATA_GOVERNANCE,
            description="Test policy description",
            created_by="admin"
        )
        
        assert policy.is_active()
        assert policy.applies_to_resource("dataset")  # No specific resources, applies to all
        assert policy.applies_to_user(["any_role"])  # No specific roles, applies to all
        
        # Test policy with specific applicability
        policy.applicable_resources = ["dataset", "model"]
        policy.applicable_roles = ["data_scientist", "ml_engineer"]
        
        assert policy.applies_to_resource("dataset")
        assert not policy.applies_to_resource("server")
        assert policy.applies_to_user(["data_scientist", "other_role"])
        assert not policy.applies_to_user(["analyst"])
    
    def test_risk_assessment_calculations(self):
        """Test risk assessment calculations."""
        assessment = RiskAssessment(
            assessor="analyst",
            risk_category="Data Breach",
            risk_description="Potential data exposure",
            likelihood=RiskLevel.HIGH,
            impact=RiskLevel.VERY_HIGH,
            overall_risk=RiskLevel.CRITICAL,
            affected_assets=["database", "api"]
        )
        
        # Test risk score calculation
        risk_score = assessment.calculate_risk_score()
        assert risk_score == 20  # HIGH (4) * VERY_HIGH (5) = 20
        
        # Test review needed
        assessment.review_date = datetime.utcnow() - timedelta(days=1)
        assert assessment.needs_review()
        
        assessment.review_date = datetime.utcnow() + timedelta(days=1)
        assert not assessment.needs_review()
    
    def test_change_request_approval_workflow(self):
        """Test change request approval workflow."""
        change_request = ChangeRequest(
            title="Test Change",
            description="Test change description",
            change_type="configuration",
            requested_by="requester",
            approvers=["approver1", "approver2"]
        )
        
        assert change_request.approval_status == ApprovalStatus.PENDING
        
        # First approval
        change_request.add_approval("approver1", "approved", "Looks good")
        assert change_request.approval_status == ApprovalStatus.PENDING  # Still needs more approvals
        assert len(change_request.approval_history) == 1
        
        # Second approval - should complete
        change_request.add_approval("approver2", "approved", "Security approved")
        assert change_request.approval_status == ApprovalStatus.APPROVED
        assert len(change_request.approval_history) == 2
        
        # Test rejection
        change_request2 = ChangeRequest(
            title="Test Rejection",
            description="Test rejection",
            change_type="policy",
            requested_by="requester",
            approvers=["approver1"]
        )
        
        change_request2.add_approval("approver1", "rejected", "Security concerns")
        assert change_request2.approval_status == ApprovalStatus.REJECTED
    
    def test_compliance_metric_status(self):
        """Test compliance metric status calculation."""
        # Compliant metric
        metric = ComplianceMetric(
            metric_name="Test Metric",
            framework=ComplianceFramework.SOC2,
            control_id="CC1.1",
            current_value=95.0,
            target_value=90.0,
            threshold_warning=80.0,
            threshold_critical=70.0
        )
        
        assert metric.get_compliance_status() == "compliant"
        
        # At risk metric
        metric.current_value = 85.0
        assert metric.get_compliance_status() == "at_risk"
        
        # Non-compliant metric
        metric.current_value = 75.0
        assert metric.get_compliance_status() == "non_compliant"
        
        # Critical metric
        metric.current_value = 65.0
        assert metric.get_compliance_status() == "critical"