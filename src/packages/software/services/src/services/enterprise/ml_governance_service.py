"""ML Governance Application Service."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import pandas as pd

from pynomaly.domain.entities.model import Model
from pynomaly.infrastructure.ml_governance import (
    DeploymentStrategy,
    GovernanceStatus,
    MLGovernanceFramework,
    ModelGovernanceRecord,
    ModelStage,
)

logger = logging.getLogger(__name__)


class MLGovernanceApplicationService:
    """Application service for ML governance operations."""

    def __init__(self, governance_framework: MLGovernanceFramework):
        """Initialize the ML governance application service."""
        self.governance_framework = governance_framework

    async def onboard_model(
        self,
        model: Model,
        validation_data: pd.DataFrame,
        model_info: dict[str, Any],
        created_by: str,
        policy_name: str = "default",
    ) -> ModelGovernanceRecord:
        """Onboard a new model to the governance framework."""
        logger.info(f"Onboarding model {model.id} to governance framework")

        try:
            # Register model
            record = await self.governance_framework.register_model(
                model=model, policy_name=policy_name, created_by=created_by
            )

            # Create model card
            await self.governance_framework.create_model_card(
                record_id=record.record_id, model_info=model_info, created_by=created_by
            )

            # Run validation
            validation_results = await self.governance_framework.validate_model(
                record_id=record.record_id, validation_data=validation_data
            )

            logger.info(
                f"Model {model.id} validation completed: {validation_results['passed']}"
            )

            # Run compliance check if validation passed
            if validation_results["passed"]:
                compliance_results = (
                    await self.governance_framework.run_compliance_check(
                        record_id=record.record_id
                    )
                )
                logger.info(
                    f"Model {model.id} compliance check completed: {compliance_results['overall_score']}"
                )

            return record

        except Exception as e:
            logger.error(f"Failed to onboard model {model.id}: {str(e)}")
            raise

    async def request_model_approval(
        self, record_id: UUID, requested_by: str
    ) -> list[dict[str, Any]]:
        """Request approval for model deployment."""
        logger.info(f"Requesting approval for governance record {record_id}")

        try:
            record = self.governance_framework.get_model_governance_record(record_id)
            if not record:
                raise ValueError(f"Governance record {record_id} not found")

            # Check if model is ready for approval
            if record.status not in [
                GovernanceStatus.APPROVED,
                GovernanceStatus.UNDER_REVIEW,
            ]:
                raise ValueError(
                    f"Model not ready for approval. Current status: {record.status}"
                )

            # Request approvals from all required roles
            approval_requests = []

            # Get required approval roles from policy
            policy = self.governance_framework.policies.get("default")
            if policy:
                for role in policy.approval_roles:
                    approval_request = await self.governance_framework.request_approval(
                        record_id=record_id,
                        approver_role=role,
                        requested_by=requested_by,
                    )
                    approval_requests.append(approval_request)

            logger.info(
                f"Approval requests created for model {record.model_id}: {len(approval_requests)} requests"
            )
            return approval_requests

        except Exception as e:
            logger.error(f"Failed to request approval for record {record_id}: {str(e)}")
            raise

    async def approve_model_deployment(
        self, record_id: UUID, approval_id: str, approver: str, comments: str = ""
    ) -> dict[str, Any]:
        """Approve model for deployment."""
        logger.info(f"Processing approval {approval_id} for record {record_id}")

        try:
            approval = await self.governance_framework.approve_model(
                record_id=record_id,
                approval_id=approval_id,
                approver=approver,
                comments=comments,
            )

            logger.info(f"Model approval processed by {approver}")
            return approval

        except Exception as e:
            logger.error(f"Failed to process approval {approval_id}: {str(e)}")
            raise

    async def deploy_model_to_stage(
        self,
        record_id: UUID,
        target_stage: ModelStage,
        deployment_strategy: DeploymentStrategy | None = None,
    ) -> dict[str, Any]:
        """Deploy model to specified stage."""
        logger.info(f"Deploying model from record {record_id} to {target_stage.value}")

        try:
            deployment_result = await self.governance_framework.deploy_model(
                record_id=record_id,
                target_stage=target_stage,
                deployment_strategy=deployment_strategy,
            )

            logger.info(f"Model deployment completed: {deployment_result['status']}")
            return deployment_result

        except Exception as e:
            logger.error(f"Failed to deploy model from record {record_id}: {str(e)}")
            raise

    async def promote_model_through_stages(
        self, record_id: UUID, auto_approve: bool = False
    ) -> list[dict[str, Any]]:
        """Promote model through development -> staging -> production stages."""
        logger.info(f"Promoting model from record {record_id} through stages")

        try:
            record = self.governance_framework.get_model_governance_record(record_id)
            if not record:
                raise ValueError(f"Governance record {record_id} not found")

            promotion_results = []

            # Define promotion path
            stage_path = [
                (ModelStage.STAGING, DeploymentStrategy.BLUE_GREEN),
                (ModelStage.PRODUCTION, DeploymentStrategy.CANARY),
            ]

            for target_stage, deployment_strategy in stage_path:
                if record.stage.value >= target_stage.value:
                    continue  # Skip if already at or past this stage

                # Check if approvals are needed
                if not auto_approve and target_stage == ModelStage.PRODUCTION:
                    # Request approvals for production deployment
                    approval_requests = await self.request_model_approval(
                        record_id=record_id, requested_by="system"
                    )

                    # For demo purposes, auto-approve all requests
                    for approval_request in approval_requests:
                        await self.approve_model_deployment(
                            record_id=record_id,
                            approval_id=approval_request["approval_id"],
                            approver="system_admin",
                            comments="Auto-approved for promotion",
                        )

                # Deploy to stage
                deployment_result = await self.deploy_model_to_stage(
                    record_id=record_id,
                    target_stage=target_stage,
                    deployment_strategy=deployment_strategy,
                )

                promotion_results.append(
                    {"stage": target_stage.value, "deployment": deployment_result}
                )

                # Update record reference
                record = self.governance_framework.get_model_governance_record(
                    record_id
                )

            logger.info(
                f"Model promotion completed through {len(promotion_results)} stages"
            )
            return promotion_results

        except Exception as e:
            logger.error(f"Failed to promote model from record {record_id}: {str(e)}")
            raise

    async def run_governance_audit(self, record_id: UUID) -> dict[str, Any]:
        """Run comprehensive governance audit."""
        logger.info(f"Running governance audit for record {record_id}")

        try:
            # Generate governance report
            report = await self.governance_framework.generate_governance_report(
                record_id
            )

            # Add audit-specific information
            audit_report = {
                **report,
                "audit_timestamp": pd.Timestamp.now().isoformat(),
                "audit_findings": [],
                "recommendations": [],
            }

            # Analyze governance compliance
            findings = []
            recommendations = []

            # Check documentation completeness
            if not report["documentation_status"]["model_card_exists"]:
                findings.append("Missing model card documentation")
                recommendations.append(
                    "Create comprehensive model card with performance metrics and limitations"
                )

            # Check approval status
            if report["approval_summary"]["pending_count"] > 0:
                findings.append(
                    f"{report['approval_summary']['pending_count']} pending approvals"
                )
                recommendations.append(
                    "Follow up on pending approvals to complete governance process"
                )

            # Check compliance score
            compliance_score = report["compliance_summary"]["latest_compliance_score"]
            if compliance_score < 0.8:
                findings.append(f"Low compliance score: {compliance_score:.2f}")
                recommendations.append(
                    "Address compliance violations to improve governance score"
                )

            audit_report["audit_findings"] = findings
            audit_report["recommendations"] = recommendations
            audit_report["overall_governance_score"] = self._calculate_governance_score(
                report
            )

            logger.info(f"Governance audit completed with {len(findings)} findings")
            return audit_report

        except Exception as e:
            logger.error(
                f"Failed to run governance audit for record {record_id}: {str(e)}"
            )
            raise

    async def get_governance_dashboard(self) -> dict[str, Any]:
        """Get comprehensive governance dashboard."""
        logger.info("Generating governance dashboard")

        try:
            dashboard_data = await self.governance_framework.get_governance_dashboard()

            # Add additional dashboard metrics
            dashboard_data["governance_health_score"] = (
                self._calculate_overall_governance_health()
            )
            dashboard_data["recommendations"] = (
                self._generate_governance_recommendations(dashboard_data)
            )

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate governance dashboard: {str(e)}")
            raise

    async def bulk_compliance_check(
        self, stage: ModelStage | None = None
    ) -> dict[str, Any]:
        """Run compliance checks on multiple models."""
        logger.info(f"Running bulk compliance check for stage: {stage}")

        try:
            # Get models to check
            records = self.governance_framework.list_governance_records(stage=stage)

            compliance_results = {
                "total_models": len(records),
                "compliant_models": 0,
                "non_compliant_models": 0,
                "results": [],
            }

            # Run compliance checks concurrently
            tasks = []
            for record in records:
                task = self.governance_framework.run_compliance_check(record.record_id)
                tasks.append((record.record_id, task))

            # Execute compliance checks
            for record_id, task in tasks:
                try:
                    result = await task
                    compliance_results["results"].append(
                        {
                            "record_id": str(record_id),
                            "compliance_score": result["overall_score"],
                            "status": "compliant"
                            if result["overall_score"] >= 0.8
                            else "non_compliant",
                        }
                    )

                    if result["overall_score"] >= 0.8:
                        compliance_results["compliant_models"] += 1
                    else:
                        compliance_results["non_compliant_models"] += 1

                except Exception as e:
                    logger.error(
                        f"Compliance check failed for record {record_id}: {str(e)}"
                    )
                    compliance_results["results"].append(
                        {
                            "record_id": str(record_id),
                            "compliance_score": 0.0,
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    compliance_results["non_compliant_models"] += 1

            logger.info(
                f"Bulk compliance check completed: {compliance_results['compliant_models']}/{compliance_results['total_models']} compliant"
            )
            return compliance_results

        except Exception as e:
            logger.error(f"Failed to run bulk compliance check: {str(e)}")
            raise

    def _calculate_governance_score(self, report: dict[str, Any]) -> float:
        """Calculate overall governance score."""
        score_components = []

        # Compliance score (40% weight)
        compliance_score = report["compliance_summary"]["latest_compliance_score"]
        score_components.append(compliance_score * 0.4)

        # Approval completion (30% weight)
        total_approvals = report["approval_summary"]["total_approvals"]
        approved_count = report["approval_summary"]["approved_count"]
        approval_score = approved_count / total_approvals if total_approvals > 0 else 0
        score_components.append(approval_score * 0.3)

        # Documentation completeness (20% weight)
        doc_score = 0
        if report["documentation_status"]["model_card_exists"]:
            doc_score += 0.5
        if report["documentation_status"]["data_sheet_exists"]:
            doc_score += 0.5
        score_components.append(doc_score * 0.2)

        # Deployment success (10% weight)
        deployment_score = (
            1.0 if report["deployment_summary"]["deployment_count"] > 0 else 0
        )
        score_components.append(deployment_score * 0.1)

        return sum(score_components)

    def _calculate_overall_governance_health(self) -> float:
        """Calculate overall governance health score."""
        records = list(self.governance_framework.records.values())
        if not records:
            return 1.0

        approved_count = len(
            [r for r in records if r.status == GovernanceStatus.APPROVED]
        )
        return approved_count / len(records)

    def _generate_governance_recommendations(
        self, dashboard_data: dict[str, Any]
    ) -> list[str]:
        """Generate governance recommendations based on dashboard data."""
        recommendations = []

        # Check compliance
        non_compliant = dashboard_data["compliance_overview"]["non_compliant_models"]
        if non_compliant > 0:
            recommendations.append(
                f"Address compliance issues for {non_compliant} non-compliant models"
            )

        # Check pending approvals
        pending = dashboard_data["approval_overview"]["pending_approvals"]
        if pending > 0:
            recommendations.append(f"Follow up on {pending} pending approvals")

        # Check governance health
        health_score = dashboard_data.get("governance_health_score", 0)
        if health_score < 0.8:
            recommendations.append(
                "Improve overall governance health by completing approval processes"
            )

        return recommendations
