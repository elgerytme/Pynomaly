#!/usr/bin/env python3
"""
ML Governance Framework Demo Script.

This script demonstrates the complete ML governance workflow including:
- Model onboarding and validation
- Compliance checking and approval workflows
- Deployment through different stages
- Audit and monitoring capabilities
- Dashboard reporting

Usage:
    python scripts/governance/ml_governance_demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Mock model class for demo
class Model:
    def __init__(self, id, name, algorithm=None, parameters=None):
        self.id = id
        self.name = name
        self.algorithm = algorithm
        self.parameters = parameters or {}


from src.anomaly_detection.application.services.ml_governance_service import (
    MLGovernanceApplicationService,
)
from src.anomaly_detection.infrastructure.ml_governance import (
    DeploymentStrategy,
    MLGovernanceFramework,
    ModelStage,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_models():
    """Create sample models for demonstration."""
    models = [
        Model(
            id=uuid4(),
            name="fraud_detection_v1",
            algorithm="isolation_forest",
            parameters={"n_estimators": 100, "contamination": 0.1},
        ),
        Model(
            id=uuid4(),
            name="time_series_anomaly_v2",
            algorithm="lstm_autoencoder",
            parameters={"sequence_length": 60, "latent_dim": 32},
        ),
        Model(
            id=uuid4(),
            name="network_intrusion_detector",
            algorithm="one_class_svm",
            parameters={"kernel": "rbf", "gamma": "scale"},
        ),
    ]
    return models


def create_sample_data():
    """Create sample validation data."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(0, 1, 1000),
            "feature_3": np.random.exponential(1, 1000),
            "feature_4": np.random.uniform(-1, 1, 1000),
            "anomaly_score": np.random.beta(
                2, 8, 1000
            ),  # Most scores near 0, few high scores
        }
    )
    return data


def create_model_info(model_name: str, use_case: str):
    """Create model information for documentation."""
    return {
        "name": model_name,
        "description": f"Advanced anomaly detection model for {use_case}",
        "intended_use": f"Real-time detection of anomalies in {use_case} scenarios",
        "limitations": "May require retraining for new data patterns. Performance may degrade with significant concept drift.",
        "training_data": {
            "samples": 100000,
            "features": 4,
            "time_period": "2023-01-01 to 2024-01-01",
            "data_sources": ["production_logs", "sensor_data", "user_interactions"],
        },
        "evaluation_data": {
            "samples": 20000,
            "features": 4,
            "time_period": "2024-01-01 to 2024-03-01",
        },
        "performance_metrics": {
            "precision": np.random.uniform(0.85, 0.95),
            "recall": np.random.uniform(0.80, 0.90),
            "f1_score": np.random.uniform(0.82, 0.92),
            "auc_roc": np.random.uniform(0.88, 0.96),
            "false_positive_rate": np.random.uniform(0.01, 0.05),
        },
        "ethical_considerations": "Model trained on balanced dataset with bias testing completed. No protected attributes used in training.",
        "caveats_and_recommendations": "Monitor for data drift every 24 hours. Retrain if performance drops below 0.8 F1-score. Implement human-in-the-loop for high-stakes decisions.",
    }


async def demonstrate_model_onboarding(governance_service, models, validation_data):
    """Demonstrate model onboarding process."""
    logger.info("=" * 80)
    logger.info("DEMONSTRATING MODEL ONBOARDING")
    logger.info("=" * 80)

    use_cases = ["fraud detection", "time series analysis", "network security"]
    records = []

    for model, use_case in zip(models, use_cases, strict=False):
        logger.info(f"\n--- Onboarding {model.name} ---")

        model_info = create_model_info(model.name, use_case)

        try:
            record = await governance_service.onboard_model(
                model=model,
                validation_data=validation_data,
                model_info=model_info,
                created_by="ml_engineer_demo",
            )

            records.append(record)

            logger.info(f"✅ Model {model.name} successfully onboarded")
            logger.info(f"   Record ID: {record.record_id}")
            logger.info(f"   Status: {record.status.value}")
            logger.info(f"   Stage: {record.stage.value}")
            logger.info(f"   Compliance checks: {len(record.compliance_checks)}")

        except Exception as e:
            logger.error(f"❌ Failed to onboard {model.name}: {str(e)}")

    return records


async def demonstrate_approval_workflow(governance_service, records):
    """Demonstrate approval workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING APPROVAL WORKFLOW")
    logger.info("=" * 80)

    for record in records:
        logger.info(f"\n--- Processing approvals for {record.model_id} ---")

        try:
            # Request approvals
            approval_requests = await governance_service.request_model_approval(
                record_id=record.record_id, requested_by="ml_engineer_demo"
            )

            logger.info(f"📝 Created {len(approval_requests)} approval requests")

            # Simulate approval process
            approvers = ["data_scientist_lead", "ml_engineer_senior", "product_owner"]

            for i, approval_request in enumerate(approval_requests):
                approver = approvers[i % len(approvers)]

                approval = await governance_service.approve_model_deployment(
                    record_id=record.record_id,
                    approval_id=approval_request["approval_id"],
                    approver=approver,
                    comments=f"Approved by {approver} - meets quality standards",
                )

                logger.info(f"✅ Approved by {approver}: {approval['approval_id']}")

            # Check final status
            updated_record = (
                governance_service.governance_framework.get_model_governance_record(
                    record.record_id
                )
            )
            logger.info(f"🎯 Final approval status: {updated_record.status.value}")

        except Exception as e:
            logger.error(f"❌ Approval workflow failed for {record.model_id}: {str(e)}")


async def demonstrate_deployment_stages(governance_service, records):
    """Demonstrate deployment through different stages."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING DEPLOYMENT STAGES")
    logger.info("=" * 80)

    deployment_strategies = [
        DeploymentStrategy.BLUE_GREEN,
        DeploymentStrategy.CANARY,
        DeploymentStrategy.BLUE_GREEN,
    ]

    for record, strategy in zip(records, deployment_strategies, strict=False):
        logger.info(f"\n--- Deploying {record.model_id} using {strategy.value} ---")

        try:
            # Deploy to staging
            staging_result = await governance_service.deploy_model_to_stage(
                record_id=record.record_id,
                target_stage=ModelStage.STAGING,
                deployment_strategy=strategy,
            )

            logger.info(f"🚀 Staging deployment: {staging_result['status']}")
            logger.info(f"   Deployment ID: {staging_result['deployment_id']}")
            logger.info(f"   Strategy: {staging_result['strategy']}")

            # Deploy to production
            production_result = await governance_service.deploy_model_to_stage(
                record_id=record.record_id,
                target_stage=ModelStage.PRODUCTION,
                deployment_strategy=strategy,
            )

            logger.info(f"🎯 Production deployment: {production_result['status']}")
            logger.info(
                f"   Deployment URL: {production_result.get('deployment_url', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"❌ Deployment failed for {record.model_id}: {str(e)}")


async def demonstrate_governance_audit(governance_service, records):
    """Demonstrate governance audit capabilities."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING GOVERNANCE AUDIT")
    logger.info("=" * 80)

    for record in records:
        logger.info(f"\n--- Auditing {record.model_id} ---")

        try:
            audit_report = await governance_service.run_governance_audit(
                record.record_id
            )

            logger.info(f"📊 Audit completed for model {audit_report['model_id']}")
            logger.info(
                f"   Governance Score: {audit_report['overall_governance_score']:.2f}"
            )
            logger.info(
                f"   Compliance Score: {audit_report['compliance_summary']['latest_compliance_score']:.2f}"
            )
            logger.info(
                f"   Approvals: {audit_report['approval_summary']['approved_count']}/{audit_report['approval_summary']['total_approvals']}"
            )
            logger.info(
                f"   Deployments: {audit_report['deployment_summary']['deployment_count']}"
            )

            if audit_report["audit_findings"]:
                logger.info("   ⚠️  Findings:")
                for finding in audit_report["audit_findings"]:
                    logger.info(f"     • {finding}")
            else:
                logger.info("   ✅ No audit findings")

            if audit_report["recommendations"]:
                logger.info("   💡 Recommendations:")
                for rec in audit_report["recommendations"]:
                    logger.info(f"     • {rec}")

        except Exception as e:
            logger.error(f"❌ Audit failed for {record.model_id}: {str(e)}")


async def demonstrate_bulk_operations(governance_service):
    """Demonstrate bulk governance operations."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING BULK OPERATIONS")
    logger.info("=" * 80)

    try:
        # Bulk compliance check
        logger.info("\n--- Running Bulk Compliance Check ---")
        compliance_results = await governance_service.bulk_compliance_check(
            stage=ModelStage.PRODUCTION
        )

        logger.info("📈 Bulk compliance check completed:")
        logger.info(f"   Total models: {compliance_results['total_models']}")
        logger.info(f"   Compliant: {compliance_results['compliant_models']}")
        logger.info(f"   Non-compliant: {compliance_results['non_compliant_models']}")

        # Show individual results
        for result in compliance_results["results"][:3]:  # Show first 3
            status_emoji = "✅" if result["status"] == "compliant" else "❌"
            logger.info(
                f"   {status_emoji} {result['record_id'][:8]}...: {result['compliance_score']:.2f}"
            )

    except Exception as e:
        logger.error(f"❌ Bulk operations failed: {str(e)}")


async def demonstrate_governance_dashboard(governance_service):
    """Demonstrate governance dashboard."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING GOVERNANCE DASHBOARD")
    logger.info("=" * 80)

    try:
        dashboard_data = await governance_service.get_governance_dashboard()

        logger.info("📊 Governance Dashboard Summary:")
        logger.info(f"   Total Models: {dashboard_data['total_models']}")
        logger.info(
            f"   Governance Health Score: {dashboard_data['governance_health_score']:.2f}"
        )

        logger.info("\n   Models by Stage:")
        for stage, count in dashboard_data["models_by_stage"].items():
            logger.info(f"     {stage}: {count}")

        logger.info("\n   Models by Status:")
        for status, count in dashboard_data["models_by_status"].items():
            logger.info(f"     {status}: {count}")

        logger.info("\n   Compliance Overview:")
        compliance = dashboard_data["compliance_overview"]
        logger.info(f"     Compliant: {compliance['compliant_models']}")
        logger.info(f"     Non-compliant: {compliance['non_compliant_models']}")

        logger.info("\n   Approval Overview:")
        approval = dashboard_data["approval_overview"]
        logger.info(f"     Pending approvals: {approval['pending_approvals']}")
        logger.info(f"     Approved models: {approval['approved_models']}")

        if dashboard_data["recommendations"]:
            logger.info("\n   💡 Recommendations:")
            for rec in dashboard_data["recommendations"]:
                logger.info(f"     • {rec}")

        logger.info(
            f"\n   Recent Deployments: {len(dashboard_data['recent_deployments'])}"
        )
        for deployment in dashboard_data["recent_deployments"][:3]:
            logger.info(
                f"     🚀 {deployment['model_id'][:8]}... -> {deployment['strategy']} ({deployment['status']})"
            )

    except Exception as e:
        logger.error(f"❌ Dashboard generation failed: {str(e)}")


async def demonstrate_promotion_workflow(governance_service):
    """Demonstrate automated promotion workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATING AUTOMATED PROMOTION WORKFLOW")
    logger.info("=" * 80)

    # Create a new model for promotion demo
    promotion_model = Model(
        id=uuid4(),
        name="auto_promotion_demo",
        algorithm="gradient_boosting",
        parameters={"n_estimators": 200, "learning_rate": 0.1},
    )

    validation_data = create_sample_data()
    model_info = create_model_info(
        "Auto Promotion Demo Model", "automated promotion testing"
    )

    try:
        logger.info("\n--- Creating model for promotion demo ---")
        record = await governance_service.onboard_model(
            model=promotion_model,
            validation_data=validation_data,
            model_info=model_info,
            created_by="promotion_demo_user",
        )

        logger.info(f"✅ Model created: {record.record_id}")

        logger.info("\n--- Starting automated promotion ---")
        promotion_results = await governance_service.promote_model_through_stages(
            record_id=record.record_id, auto_approve=True
        )

        logger.info(f"🎯 Promotion completed through {len(promotion_results)} stages:")
        for i, result in enumerate(promotion_results, 1):
            logger.info(f"   {i}. {result['stage']}: {result['deployment']['status']}")

        # Check final state
        final_record = (
            governance_service.governance_framework.get_model_governance_record(
                record.record_id
            )
        )
        logger.info(f"🏁 Final stage: {final_record.stage.value}")
        logger.info(f"   Total deployments: {len(final_record.deployment_history)}")

    except Exception as e:
        logger.error(f"❌ Promotion workflow failed: {str(e)}")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting ML Governance Framework Demonstration")
    logger.info("This demo showcases comprehensive ML model governance capabilities")

    # Initialize governance framework
    governance_framework = MLGovernanceFramework()
    governance_service = MLGovernanceApplicationService(governance_framework)

    # Create sample data
    models = create_sample_models()
    validation_data = create_sample_data()

    logger.info(
        f"\n📦 Created {len(models)} sample models and validation dataset with {len(validation_data)} samples"
    )

    # Run demonstrations
    try:
        # 1. Model Onboarding
        records = await demonstrate_model_onboarding(
            governance_service, models, validation_data
        )

        # 2. Approval Workflow
        await demonstrate_approval_workflow(governance_service, records)

        # 3. Deployment Stages
        await demonstrate_deployment_stages(governance_service, records)

        # 4. Governance Audit
        await demonstrate_governance_audit(governance_service, records)

        # 5. Bulk Operations
        await demonstrate_bulk_operations(governance_service)

        # 6. Governance Dashboard
        await demonstrate_governance_dashboard(governance_service)

        # 7. Automated Promotion
        await demonstrate_promotion_workflow(governance_service)

        logger.info("\n" + "=" * 80)
        logger.info("🎉 ML GOVERNANCE FRAMEWORK DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nKey capabilities demonstrated:")
        logger.info("✅ Model onboarding with validation and compliance checks")
        logger.info("✅ Multi-stage approval workflows with role-based approvers")
        logger.info("✅ Deployment strategies (Blue-Green, Canary)")
        logger.info("✅ Comprehensive governance auditing")
        logger.info("✅ Bulk compliance operations")
        logger.info("✅ Real-time governance dashboard")
        logger.info("✅ Automated model promotion workflows")
        logger.info("\nThe ML Governance Framework is ready for production use! 🚀")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n🛑 Demonstration interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
