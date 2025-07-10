"""Integration tests for ML Governance Framework."""

import asyncio

import pandas as pd
import pytest

from pynomaly.application.services.ml_governance_service import (
    MLGovernanceApplicationService,
)
from pynomaly.domain.entities.model import Model, ModelType
from pynomaly.infrastructure.ml_governance import (
    DeploymentStrategy,
    GovernanceStatus,
    MLGovernanceFramework,
    ModelStage,
)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return Model(
        name="test_anomaly_detector",
        description="Test anomaly detection model using Isolation Forest",
        model_type=ModelType.UNSUPERVISED,
        algorithm_family="isolation_forest",
        created_by="test_user",
        tags=["test", "anomaly_detection"],
        use_cases=["fraud_detection"],
        data_requirements={"features": 3, "format": "numerical"},
    )


@pytest.fixture
def sample_validation_data():
    """Create sample validation data."""
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5] * 20,
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            "anomaly_score": [0.1, 0.15, 0.8, 0.12, 0.9] * 20,
        }
    )


@pytest.fixture
def governance_framework():
    """Create governance framework for testing."""
    return MLGovernanceFramework()


@pytest.fixture
def governance_service(governance_framework):
    """Create governance application service for testing."""
    return MLGovernanceApplicationService(governance_framework)


@pytest.mark.asyncio
async def test_model_onboarding_complete_workflow(
    governance_service, sample_model, sample_validation_data
):
    """Test complete model onboarding workflow."""
    model_info = {
        "name": "Test Anomaly Detector",
        "description": "A test anomaly detection model using Isolation Forest",
        "intended_use": "Detecting anomalies in time series data",
        "limitations": "May not perform well on highly seasonal data",
        "training_data": {"samples": 10000, "features": 2},
        "evaluation_data": {"samples": 2000, "features": 2},
        "performance_metrics": {"precision": 0.85, "recall": 0.82, "f1_score": 0.835},
        "ethical_considerations": "No bias concerns identified",
        "caveats_and_recommendations": "Monitor for data drift in production",
    }

    # Onboard model
    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="test_user",
    )

    # Verify record creation
    assert record is not None
    assert record.model_id == sample_model.id
    assert record.stage == ModelStage.DEVELOPMENT
    assert record.created_by == "test_user"

    # Verify model card creation
    assert record.model_card is not None
    assert record.model_card.model_name == "Test Anomaly Detector"
    assert record.model_card.description == model_info["description"]

    # Verify validation and compliance checks
    assert len(record.compliance_checks) >= 1

    # Check if validation passed (should pass with good metrics)
    validation_check = next(
        (c for c in record.compliance_checks if c["type"] == "validation"), None
    )
    assert validation_check is not None
    assert validation_check["results"]["passed"] is True


@pytest.mark.asyncio
async def test_model_approval_workflow(
    governance_service, sample_model, sample_validation_data
):
    """Test model approval workflow."""
    model_info = {
        "name": "Test Model",
        "description": "Test model for approval workflow",
        "intended_use": "Testing",
        "limitations": "Test only",
        "training_data": {},
        "evaluation_data": {},
        "performance_metrics": {"accuracy": 0.9},
        "ethical_considerations": "None",
        "caveats_and_recommendations": "Test only",
    }

    # Onboard model
    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="test_user",
    )

    # Request approvals
    approval_requests = await governance_service.request_model_approval(
        record_id=record.record_id, requested_by="test_user"
    )

    # Verify approval requests created
    assert len(approval_requests) > 0

    # Approve requests
    for approval_request in approval_requests:
        approval = await governance_service.approve_model_deployment(
            record_id=record.record_id,
            approval_id=approval_request["approval_id"],
            approver="test_approver",
            comments="Approved for testing",
        )
        assert approval["status"] == "approved"

    # Verify model is approved
    updated_record = (
        governance_service.governance_framework.get_model_governance_record(
            record.record_id
        )
    )
    assert updated_record.status == GovernanceStatus.APPROVED


@pytest.mark.asyncio
async def test_model_deployment_stages(
    governance_service, sample_model, sample_validation_data
):
    """Test model deployment through different stages."""
    model_info = {
        "name": "Deployment Test Model",
        "description": "Test model for deployment",
        "intended_use": "Testing deployment workflow",
        "limitations": "Test environment only",
        "training_data": {},
        "evaluation_data": {},
        "performance_metrics": {"f1_score": 0.88},
        "ethical_considerations": "None for testing",
        "caveats_and_recommendations": "Monitor performance",
    }

    # Onboard and approve model
    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="test_user",
    )

    # Auto-approve for testing
    approval_requests = await governance_service.request_model_approval(
        record_id=record.record_id, requested_by="test_user"
    )

    for approval_request in approval_requests:
        await governance_service.approve_model_deployment(
            record_id=record.record_id,
            approval_id=approval_request["approval_id"],
            approver="test_approver",
        )

    # Deploy to staging
    staging_deployment = await governance_service.deploy_model_to_stage(
        record_id=record.record_id,
        target_stage=ModelStage.STAGING,
        deployment_strategy=DeploymentStrategy.BLUE_GREEN,
    )

    assert staging_deployment["status"] == "completed"
    assert staging_deployment["strategy"] == "blue_green"

    # Verify stage update
    updated_record = (
        governance_service.governance_framework.get_model_governance_record(
            record.record_id
        )
    )
    assert updated_record.stage == ModelStage.STAGING

    # Deploy to production
    production_deployment = await governance_service.deploy_model_to_stage(
        record_id=record.record_id,
        target_stage=ModelStage.PRODUCTION,
        deployment_strategy=DeploymentStrategy.CANARY,
    )

    assert production_deployment["status"] == "completed"
    assert production_deployment["strategy"] == "canary"

    # Verify final stage
    final_record = governance_service.governance_framework.get_model_governance_record(
        record.record_id
    )
    assert final_record.stage == ModelStage.PRODUCTION


@pytest.mark.asyncio
async def test_model_promotion_through_stages(
    governance_service, sample_model, sample_validation_data
):
    """Test automated model promotion through all stages."""
    model_info = {
        "name": "Promotion Test Model",
        "description": "Test model for stage promotion",
        "intended_use": "Testing automated promotion",
        "limitations": "Test only",
        "training_data": {},
        "evaluation_data": {},
        "performance_metrics": {"precision": 0.9, "recall": 0.85},
        "ethical_considerations": "None",
        "caveats_and_recommendations": "Monitor closely",
    }

    # Onboard model
    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="test_user",
    )

    # Promote through stages with auto-approval
    promotion_results = await governance_service.promote_model_through_stages(
        record_id=record.record_id, auto_approve=True
    )

    # Verify promotion results
    assert len(promotion_results) == 2  # staging and production
    assert promotion_results[0]["stage"] == "staging"
    assert promotion_results[1]["stage"] == "production"

    # Verify final state
    final_record = governance_service.governance_framework.get_model_governance_record(
        record.record_id
    )
    assert final_record.stage == ModelStage.PRODUCTION
    assert len(final_record.deployment_history) == 2


@pytest.mark.asyncio
async def test_governance_audit(
    governance_service, sample_model, sample_validation_data
):
    """Test governance audit functionality."""
    model_info = {
        "name": "Audit Test Model",
        "description": "Test model for governance audit",
        "intended_use": "Testing audit capabilities",
        "limitations": "Audit testing only",
        "training_data": {"samples": 5000},
        "evaluation_data": {"samples": 1000},
        "performance_metrics": {"auc_roc": 0.92},
        "ethical_considerations": "Bias testing completed",
        "caveats_and_recommendations": "Regular monitoring required",
    }

    # Onboard model
    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="test_user",
    )

    # Run governance audit
    audit_report = await governance_service.run_governance_audit(record.record_id)

    # Verify audit report structure
    assert "audit_timestamp" in audit_report
    assert "audit_findings" in audit_report
    assert "recommendations" in audit_report
    assert "overall_governance_score" in audit_report

    # Verify governance score calculation
    governance_score = audit_report["overall_governance_score"]
    assert 0.0 <= governance_score <= 1.0

    # Verify compliance summary
    assert "compliance_summary" in audit_report
    assert "approval_summary" in audit_report
    assert "documentation_status" in audit_report


@pytest.mark.asyncio
async def test_bulk_compliance_check(governance_service, sample_validation_data):
    """Test bulk compliance checking."""
    # Create multiple models
    models = []
    records = []

    for i in range(3):
        model = Model(
            name=f"bulk_test_model_{i}",
            description=f"Bulk test model {i}",
            model_type=ModelType.UNSUPERVISED,
            algorithm_family="isolation_forest",
            created_by="bulk_test_user",
        )
        models.append(model)

        model_info = {
            "name": f"Bulk Test Model {i}",
            "description": f"Test model {i} for bulk compliance",
            "intended_use": "Bulk testing",
            "limitations": "Test only",
            "training_data": {},
            "evaluation_data": {},
            "performance_metrics": {"f1_score": 0.8 + (i * 0.05)},
            "ethical_considerations": "None",
            "caveats_and_recommendations": "Testing",
        }

        record = await governance_service.onboard_model(
            model=model,
            validation_data=sample_validation_data,
            model_info=model_info,
            created_by="bulk_test_user",
        )
        records.append(record)

    # Run bulk compliance check
    compliance_results = await governance_service.bulk_compliance_check()

    # Verify results
    assert compliance_results["total_models"] >= 3
    assert len(compliance_results["results"]) >= 3
    assert (
        compliance_results["compliant_models"]
        + compliance_results["non_compliant_models"]
        >= 3
    )

    # Verify individual results
    for result in compliance_results["results"]:
        assert "record_id" in result
        assert "compliance_score" in result
        assert "status" in result


@pytest.mark.asyncio
async def test_governance_dashboard(
    governance_service, sample_model, sample_validation_data
):
    """Test governance dashboard data generation."""
    # Create a model and go through governance workflow
    model_info = {
        "name": "Dashboard Test Model",
        "description": "Test model for dashboard",
        "intended_use": "Dashboard testing",
        "limitations": "Test only",
        "training_data": {},
        "evaluation_data": {},
        "performance_metrics": {"accuracy": 0.93},
        "ethical_considerations": "None",
        "caveats_and_recommendations": "Monitor",
    }

    record = await governance_service.onboard_model(
        model=sample_model,
        validation_data=sample_validation_data,
        model_info=model_info,
        created_by="dashboard_test_user",
    )

    # Get dashboard data
    dashboard_data = await governance_service.get_governance_dashboard()

    # Verify dashboard structure
    assert "total_models" in dashboard_data
    assert "models_by_stage" in dashboard_data
    assert "models_by_status" in dashboard_data
    assert "compliance_overview" in dashboard_data
    assert "approval_overview" in dashboard_data
    assert "recent_deployments" in dashboard_data
    assert "governance_health_score" in dashboard_data
    assert "recommendations" in dashboard_data

    # Verify data consistency
    assert dashboard_data["total_models"] >= 1
    assert isinstance(dashboard_data["governance_health_score"], float)
    assert 0.0 <= dashboard_data["governance_health_score"] <= 1.0


@pytest.mark.asyncio
async def test_governance_record_retrieval(
    governance_framework, sample_model, sample_validation_data
):
    """Test governance record retrieval and filtering."""
    # Register multiple models
    record1 = await governance_framework.register_model(
        sample_model, created_by="user1"
    )

    model2 = Model(
        name="model2",
        description="Second test model",
        model_type=ModelType.SUPERVISED,
        algorithm_family="random_forest",
        created_by="user2",
    )
    record2 = await governance_framework.register_model(model2, created_by="user2")

    # Test record retrieval by ID
    retrieved_record = governance_framework.get_model_governance_record(
        record1.record_id
    )
    assert retrieved_record is not None
    assert retrieved_record.record_id == record1.record_id

    # Test listing all records
    all_records = governance_framework.list_governance_records()
    assert len(all_records) >= 2

    # Test filtering by stage
    dev_records = governance_framework.list_governance_records(
        stage=ModelStage.DEVELOPMENT
    )
    assert len(dev_records) >= 2

    # Test filtering by status
    pending_records = governance_framework.list_governance_records(
        status=GovernanceStatus.PENDING
    )
    assert len(pending_records) >= 2


@pytest.mark.asyncio
async def test_concurrent_governance_operations(
    governance_service, sample_validation_data
):
    """Test concurrent governance operations."""
    # Create multiple models for concurrent processing
    models = [
        Model(
            name=f"concurrent_model_{i}",
            description=f"Concurrent test model {i}",
            model_type=ModelType.UNSUPERVISED,
            algorithm_family="isolation_forest",
            created_by=f"concurrent_user_{i}",
        )
        for i in range(5)
    ]

    model_info_template = {
        "name": "Concurrent Test Model",
        "description": "Test model for concurrent operations",
        "intended_use": "Concurrency testing",
        "limitations": "Test only",
        "training_data": {},
        "evaluation_data": {},
        "performance_metrics": {"f1_score": 0.85},
        "ethical_considerations": "None",
        "caveats_and_recommendations": "Testing",
    }

    # Onboard models concurrently
    tasks = []
    for i, model in enumerate(models):
        model_info = {**model_info_template, "name": f"Concurrent Test Model {i}"}
        task = governance_service.onboard_model(
            model=model,
            validation_data=sample_validation_data,
            model_info=model_info,
            created_by=f"concurrent_user_{i}",
        )
        tasks.append(task)

    # Execute concurrently
    records = await asyncio.gather(*tasks)

    # Verify all models were onboarded successfully
    assert len(records) == 5
    for record in records:
        assert record is not None
        assert record.stage == ModelStage.DEVELOPMENT
        assert len(record.compliance_checks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
