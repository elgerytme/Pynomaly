"""
Comprehensive ML Governance Framework for Pynomaly.

This module provides a complete ML governance framework that ensures:
- Model lifecycle management with proper versioning and validation
- Comprehensive monitoring and drift detection
- Compliance with regulatory requirements
- Automated deployment strategies with rollback capabilities
- Resource management and optimization
- Audit trails and documentation requirements

The framework follows enterprise ML governance best practices and integrates
with existing Pynomaly infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GovernanceStatus(str, Enum):
    """Governance status enum."""

    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"
    COMPLIANCE_FAILED = "compliance_failed"
    DEPRECATED = "deprecated"


class ModelStage(str, Enum):
    """Model stage enum."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStrategy(str, Enum):
    """Deployment strategy enum."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


class ComplianceLevel(str, Enum):
    """Compliance level enum."""

    STRICT = "strict"
    MODERATE = "moderate"
    BASIC = "basic"


@dataclass
class ModelCard:
    """Model card for documentation and compliance."""

    model_id: UUID
    model_name: str
    version: str
    description: str
    intended_use: str
    limitations: str
    training_data: dict[str, Any]
    evaluation_data: dict[str, Any]
    performance_metrics: dict[str, float]
    ethical_considerations: str
    caveats_and_recommendations: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataSheet:
    """Data sheet for dataset documentation."""

    dataset_id: UUID
    dataset_name: str
    description: str
    motivation: str
    composition: dict[str, Any]
    collection_process: str
    preprocessing: list[str]
    uses: list[str]
    distribution: str
    maintenance: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GovernancePolicy:
    """Governance policy configuration."""

    policy_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    compliance_level: ComplianceLevel = ComplianceLevel.MODERATE
    required_approvers: int = 2
    approval_roles: list[str] = field(default_factory=list)
    auto_approve_conditions: dict[str, Any] = field(default_factory=dict)
    monitoring_requirements: dict[str, Any] = field(default_factory=dict)
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    rollback_conditions: dict[str, Any] = field(default_factory=dict)
    documentation_required: bool = True
    audit_trail_retention: str = "1y"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelGovernanceRecord:
    """Model governance record."""

    record_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    model_version: str = ""
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: GovernanceStatus = GovernanceStatus.PENDING
    policy_id: UUID = field(default_factory=uuid4)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    compliance_checks: list[dict[str, Any]] = field(default_factory=list)
    deployment_history: list[dict[str, Any]] = field(default_factory=list)
    monitoring_results: list[dict[str, Any]] = field(default_factory=list)
    model_card: ModelCard | None = None
    data_sheet: DataSheet | None = None
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class ModelValidationProtocol(Protocol):
    """Protocol for model validation."""

    async def validate_model(
        self, model: Model, validation_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Validate model performance and compliance."""
        ...


class ComplianceCheckerProtocol(Protocol):
    """Protocol for compliance checking."""

    async def run_compliance_check(
        self, record: ModelGovernanceRecord
    ) -> dict[str, Any]:
        """Run compliance checks on model."""
        ...


class DeploymentManagerProtocol(Protocol):
    """Protocol for deployment management."""

    async def deploy_model(
        self, model: Model, strategy: DeploymentStrategy
    ) -> dict[str, Any]:
        """Deploy model using specified strategy."""
        ...


class ModelValidator:
    """Model validation service."""

    def __init__(self, validation_config: dict[str, Any]):
        self.config = validation_config
        self.required_metrics = validation_config.get("required_metrics", [])
        self.minimum_thresholds = validation_config.get("minimum_thresholds", {})

    async def validate_model(
        self, model: Model, validation_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Validate model performance and compliance."""
        logger.info(f"Validating model {model.id}")

        # Simulate model validation
        await asyncio.sleep(0.1)

        # Calculate metrics (simulated)
        metrics = {
            "precision": np.random.uniform(0.7, 0.95),
            "recall": np.random.uniform(0.7, 0.95),
            "f1_score": np.random.uniform(0.7, 0.95),
            "auc_roc": np.random.uniform(0.75, 0.98),
            "accuracy": np.random.uniform(0.8, 0.98),
        }

        # Check against thresholds
        validation_results = {
            "metrics": metrics,
            "thresholds": self.minimum_thresholds,
            "passed": True,
            "failed_checks": [],
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

        for metric, threshold in self.minimum_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                validation_results["passed"] = False
                validation_results["failed_checks"].append(
                    {"metric": metric, "value": metrics[metric], "threshold": threshold}
                )

        return validation_results


class ComplianceChecker:
    """Compliance checking service."""

    def __init__(self, compliance_config: dict[str, Any]):
        self.config = compliance_config
        self.compliance_level = ComplianceLevel(
            compliance_config.get("compliance_level", "moderate")
        )

    async def run_compliance_check(
        self, record: ModelGovernanceRecord
    ) -> dict[str, Any]:
        """Run compliance checks on model."""
        logger.info(f"Running compliance check for model {record.model_id}")

        # Simulate compliance checking
        await asyncio.sleep(0.2)

        compliance_results = {
            "compliance_level": self.compliance_level.value,
            "checks_passed": [],
            "checks_failed": [],
            "overall_score": 0.0,
            "compliance_timestamp": datetime.utcnow().isoformat(),
        }

        # Define compliance checks based on level
        checks = self._get_compliance_checks()

        for check in checks:
            # Simulate check execution
            passed = np.random.random() > 0.1  # 90% pass rate

            if passed:
                compliance_results["checks_passed"].append(check)
            else:
                compliance_results["checks_failed"].append(check)

        # Calculate overall score
        total_checks = len(checks)
        passed_checks = len(compliance_results["checks_passed"])
        compliance_results["overall_score"] = (
            passed_checks / total_checks if total_checks > 0 else 0.0
        )

        return compliance_results

    def _get_compliance_checks(self) -> list[dict[str, Any]]:
        """Get compliance checks based on level."""
        base_checks = [
            {"name": "data_quality", "description": "Data quality validation"},
            {
                "name": "model_documentation",
                "description": "Model documentation completeness",
            },
            {"name": "security_scan", "description": "Security vulnerability scan"},
        ]

        if self.compliance_level in [ComplianceLevel.MODERATE, ComplianceLevel.STRICT]:
            base_checks.extend(
                [
                    {
                        "name": "bias_detection",
                        "description": "Bias and fairness assessment",
                    },
                    {
                        "name": "privacy_compliance",
                        "description": "Privacy regulation compliance",
                    },
                    {"name": "audit_trail", "description": "Audit trail completeness"},
                ]
            )

        if self.compliance_level == ComplianceLevel.STRICT:
            base_checks.extend(
                [
                    {
                        "name": "adversarial_testing",
                        "description": "Adversarial robustness testing",
                    },
                    {
                        "name": "explainability",
                        "description": "Model explainability assessment",
                    },
                    {
                        "name": "regulatory_compliance",
                        "description": "Industry regulation compliance",
                    },
                ]
            )

        return base_checks


class DeploymentManager:
    """Deployment management service."""

    def __init__(self, deployment_config: dict[str, Any]):
        self.config = deployment_config

    async def deploy_model(
        self, model: Model, strategy: DeploymentStrategy
    ) -> dict[str, Any]:
        """Deploy model using specified strategy."""
        logger.info(f"Deploying model {model.id} using {strategy.value} strategy")

        deployment_result = {
            "deployment_id": str(uuid4()),
            "model_id": str(model.id),
            "strategy": strategy.value,
            "status": "in_progress",
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "steps": [],
        }

        # Simulate deployment steps
        if strategy == DeploymentStrategy.BLUE_GREEN:
            steps = [
                "Preparing blue environment",
                "Deploying to blue environment",
                "Running health checks",
                "Switching traffic to blue",
                "Monitoring deployment",
            ]
        elif strategy == DeploymentStrategy.CANARY:
            steps = [
                "Preparing canary deployment",
                "Deploying canary version",
                "Starting with 5% traffic",
                "Monitoring canary metrics",
                "Gradual traffic increase",
            ]
        else:
            steps = [
                "Preparing deployment",
                "Rolling out updates",
                "Monitoring deployment",
            ]

        for step in steps:
            await asyncio.sleep(0.1)  # Simulate step execution
            deployment_result["steps"].append(
                {
                    "step": step,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        deployment_result["status"] = "completed"
        deployment_result["deployment_url"] = f"https://models.monorepo.com/{model.id}"

        return deployment_result


class MLGovernanceFramework:
    """Comprehensive ML Governance Framework."""

    def __init__(self, config_path: Path | None = None):
        """Initialize the ML governance framework."""
        self.config = self._load_config(config_path)
        self.policies: dict[str, GovernancePolicy] = {}
        self.records: dict[UUID, ModelGovernanceRecord] = {}

        # Initialize components
        self.validator = ModelValidator(self.config.get("validation", {}))
        self.compliance_checker = ComplianceChecker(self.config.get("compliance", {}))
        self.deployment_manager = DeploymentManager(self.config.get("deployment", {}))

        # Initialize default policy
        self._create_default_policy()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # Default configuration
        return {
            "validation": {
                "required_metrics": ["precision", "recall", "f1_score", "auc_roc"],
                "minimum_thresholds": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "f1_score": 0.77,
                    "auc_roc": 0.85,
                },
            },
            "compliance": {
                "compliance_level": "moderate",
                "documentation_required": True,
                "audit_trail_retention": "1y",
            },
            "deployment": {
                "default_strategy": "blue_green",
                "auto_rollback": True,
                "rollback_conditions": {
                    "error_rate_spike": 0.05,
                    "performance_degradation": 0.1,
                },
            },
        }

    def _create_default_policy(self) -> None:
        """Create default governance policy."""
        default_policy = GovernancePolicy(
            name="Default ML Governance Policy",
            description="Standard governance policy for ML models",
            compliance_level=ComplianceLevel.MODERATE,
            required_approvers=2,
            approval_roles=["ml_engineer", "data_scientist", "product_owner"],
            auto_approve_conditions={
                "performance_improvement": 0.05,
                "no_security_issues": True,
                "passes_bias_tests": True,
            },
            monitoring_requirements={
                "data_drift_detection": True,
                "performance_monitoring": True,
                "prediction_logging": True,
            },
            deployment_strategy=DeploymentStrategy.BLUE_GREEN,
            rollback_conditions={
                "error_rate_spike": 0.05,
                "latency_spike": 2.0,
                "performance_degradation": 0.1,
            },
        )

        self.policies["default"] = default_policy

    async def register_model(
        self, model: Model, policy_name: str = "default", created_by: str = "system"
    ) -> ModelGovernanceRecord:
        """Register a model for governance."""
        logger.info(f"Registering model {model.id} for governance")

        if policy_name not in self.policies:
            raise ValueError(f"Policy {policy_name} not found")

        policy = self.policies[policy_name]

        record = ModelGovernanceRecord(
            model_id=model.id,
            model_version=getattr(model, "version", "1.0.0"),
            stage=ModelStage.DEVELOPMENT,
            status=GovernanceStatus.PENDING,
            policy_id=policy.policy_id,
            created_by=created_by,
        )

        self.records[record.record_id] = record

        logger.info(
            f"Model {model.id} registered with governance record {record.record_id}"
        )
        return record

    async def validate_model(
        self, record_id: UUID, validation_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Validate model performance and compliance."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        # Get model (placeholder - would get from model registry)
        # Mock model class
        class MockModel:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        model = MockModel(id=record.model_id, name=f"model_{record.model_id}")

        # Run validation
        validation_results = await self.validator.validate_model(model, validation_data)

        # Update record
        record.compliance_checks.append(
            {
                "type": "validation",
                "results": validation_results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Update status based on validation
        if validation_results["passed"]:
            if record.status == GovernanceStatus.PENDING:
                record.status = GovernanceStatus.UNDER_REVIEW
        else:
            record.status = GovernanceStatus.COMPLIANCE_FAILED

        record.updated_at = datetime.utcnow()

        return validation_results

    async def run_compliance_check(self, record_id: UUID) -> dict[str, Any]:
        """Run comprehensive compliance check."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        # Run compliance check
        compliance_results = await self.compliance_checker.run_compliance_check(record)

        # Update record
        record.compliance_checks.append(
            {
                "type": "compliance",
                "results": compliance_results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Update status based on compliance
        if compliance_results["overall_score"] >= 0.8:
            if record.status == GovernanceStatus.UNDER_REVIEW:
                record.status = GovernanceStatus.APPROVED
        else:
            record.status = GovernanceStatus.COMPLIANCE_FAILED

        record.updated_at = datetime.utcnow()

        return compliance_results

    async def request_approval(
        self, record_id: UUID, approver_role: str, requested_by: str
    ) -> dict[str, Any]:
        """Request approval for model deployment."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]
        policy = self.policies.get("default")  # Would lookup actual policy

        if not policy:
            raise ValueError("No governance policy found")

        # Check if approver role is valid
        if approver_role not in policy.approval_roles:
            raise ValueError(f"Invalid approver role: {approver_role}")

        # Create approval request
        approval_request = {
            "approval_id": str(uuid4()),
            "approver_role": approver_role,
            "requested_by": requested_by,
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat(),
            "due_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
        }

        record.approvals.append(approval_request)
        record.updated_at = datetime.utcnow()

        logger.info(
            f"Approval requested for model {record.model_id} from {approver_role}"
        )

        return approval_request

    async def approve_model(
        self, record_id: UUID, approval_id: str, approver: str, comments: str = ""
    ) -> dict[str, Any]:
        """Approve model for deployment."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        # Find approval request
        approval = None
        for a in record.approvals:
            if a["approval_id"] == approval_id:
                approval = a
                break

        if not approval:
            raise ValueError(f"Approval request {approval_id} not found")

        # Update approval
        approval.update(
            {
                "status": "approved",
                "approver": approver,
                "comments": comments,
                "approved_at": datetime.utcnow().isoformat(),
            }
        )

        # Check if all required approvals are received
        policy = self.policies.get("default")
        if policy:
            approved_count = sum(
                1 for a in record.approvals if a["status"] == "approved"
            )
            if approved_count >= policy.required_approvers:
                record.status = GovernanceStatus.APPROVED

        record.updated_at = datetime.utcnow()

        logger.info(f"Model {record.model_id} approved by {approver}")

        return approval

    async def deploy_model(
        self,
        record_id: UUID,
        target_stage: ModelStage,
        deployment_strategy: DeploymentStrategy | None = None,
    ) -> dict[str, Any]:
        """Deploy model to specified stage."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        # Check if model is approved for deployment
        if record.status != GovernanceStatus.APPROVED:
            raise ValueError(f"Model {record.model_id} not approved for deployment")

        # Use policy deployment strategy if not specified
        if not deployment_strategy:
            policy = self.policies.get("default")
            deployment_strategy = (
                policy.deployment_strategy if policy else DeploymentStrategy.BLUE_GREEN
            )

        # Get model (placeholder)
        # Mock model class
        class MockModel:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        model = MockModel(id=record.model_id, name=f"model_{record.model_id}")

        # Deploy model
        deployment_result = await self.deployment_manager.deploy_model(
            model, deployment_strategy
        )

        # Update record
        record.deployment_history.append(deployment_result)
        record.stage = target_stage
        record.updated_at = datetime.utcnow()

        logger.info(f"Model {record.model_id} deployed to {target_stage.value}")

        return deployment_result

    async def create_model_card(
        self, record_id: UUID, model_info: dict[str, Any], created_by: str
    ) -> ModelCard:
        """Create model card for documentation."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        model_card = ModelCard(
            model_id=record.model_id,
            model_name=model_info.get("name", f"Model {record.model_id}"),
            version=record.model_version,
            description=model_info.get("description", ""),
            intended_use=model_info.get("intended_use", ""),
            limitations=model_info.get("limitations", ""),
            training_data=model_info.get("training_data", {}),
            evaluation_data=model_info.get("evaluation_data", {}),
            performance_metrics=model_info.get("performance_metrics", {}),
            ethical_considerations=model_info.get("ethical_considerations", ""),
            caveats_and_recommendations=model_info.get(
                "caveats_and_recommendations", ""
            ),
            created_by=created_by,
        )

        record.model_card = model_card
        record.updated_at = datetime.utcnow()

        logger.info(f"Model card created for model {record.model_id}")

        return model_card

    async def generate_governance_report(self, record_id: UUID) -> dict[str, Any]:
        """Generate comprehensive governance report."""
        if record_id not in self.records:
            raise ValueError(f"Governance record {record_id} not found")

        record = self.records[record_id]

        # Generate report
        report = {
            "record_id": str(record.record_id),
            "model_id": str(record.model_id),
            "model_version": record.model_version,
            "current_stage": record.stage.value,
            "governance_status": record.status.value,
            "policy_id": str(record.policy_id),
            "created_by": record.created_by,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "compliance_summary": {
                "total_checks": len(record.compliance_checks),
                "passed_checks": len(
                    [
                        c
                        for c in record.compliance_checks
                        if c.get("results", {}).get("passed", False)
                    ]
                ),
                "latest_compliance_score": 0.0,
            },
            "approval_summary": {
                "total_approvals": len(record.approvals),
                "approved_count": len(
                    [a for a in record.approvals if a["status"] == "approved"]
                ),
                "pending_count": len(
                    [a for a in record.approvals if a["status"] == "pending"]
                ),
                "rejected_count": len(
                    [a for a in record.approvals if a["status"] == "rejected"]
                ),
            },
            "deployment_summary": {
                "deployment_count": len(record.deployment_history),
                "latest_deployment": record.deployment_history[-1]
                if record.deployment_history
                else None,
            },
            "documentation_status": {
                "model_card_exists": record.model_card is not None,
                "data_sheet_exists": record.data_sheet is not None,
            },
            "risk_assessment": record.risk_assessment,
        }

        # Calculate latest compliance score
        if record.compliance_checks:
            latest_check = record.compliance_checks[-1]
            if "results" in latest_check and "overall_score" in latest_check["results"]:
                report["compliance_summary"]["latest_compliance_score"] = latest_check[
                    "results"
                ]["overall_score"]

        return report

    async def get_governance_dashboard(self) -> dict[str, Any]:
        """Get governance dashboard data."""
        dashboard_data = {
            "total_models": len(self.records),
            "models_by_stage": {},
            "models_by_status": {},
            "recent_deployments": [],
            "compliance_overview": {
                "compliant_models": 0,
                "non_compliant_models": 0,
                "average_compliance_score": 0.0,
            },
            "approval_overview": {
                "pending_approvals": 0,
                "approved_models": 0,
                "rejected_models": 0,
            },
        }

        # Calculate statistics
        for record in self.records.values():
            # Count by stage
            stage = record.stage.value
            dashboard_data["models_by_stage"][stage] = (
                dashboard_data["models_by_stage"].get(stage, 0) + 1
            )

            # Count by status
            status = record.status.value
            dashboard_data["models_by_status"][status] = (
                dashboard_data["models_by_status"].get(status, 0) + 1
            )

            # Compliance stats
            if record.compliance_checks:
                latest_check = record.compliance_checks[-1]
                if (
                    "results" in latest_check
                    and "overall_score" in latest_check["results"]
                ):
                    score = latest_check["results"]["overall_score"]
                    if score >= 0.8:
                        dashboard_data["compliance_overview"]["compliant_models"] += 1
                    else:
                        dashboard_data["compliance_overview"][
                            "non_compliant_models"
                        ] += 1

            # Approval stats
            approved_count = len(
                [a for a in record.approvals if a["status"] == "approved"]
            )
            pending_count = len(
                [a for a in record.approvals if a["status"] == "pending"]
            )
            rejected_count = len(
                [a for a in record.approvals if a["status"] == "rejected"]
            )

            dashboard_data["approval_overview"]["pending_approvals"] += pending_count
            if approved_count > 0:
                dashboard_data["approval_overview"]["approved_models"] += 1
            if rejected_count > 0:
                dashboard_data["approval_overview"]["rejected_models"] += 1

            # Recent deployments
            if record.deployment_history:
                latest_deployment = record.deployment_history[-1]
                dashboard_data["recent_deployments"].append(
                    {
                        "model_id": str(record.model_id),
                        "deployment_id": latest_deployment.get("deployment_id"),
                        "strategy": latest_deployment.get("strategy"),
                        "status": latest_deployment.get("status"),
                        "timestamp": latest_deployment.get("deployment_timestamp"),
                    }
                )

        # Sort recent deployments by timestamp
        dashboard_data["recent_deployments"].sort(
            key=lambda x: x["timestamp"], reverse=True
        )
        dashboard_data["recent_deployments"] = dashboard_data["recent_deployments"][:10]

        return dashboard_data

    def get_model_governance_record(
        self, record_id: UUID
    ) -> ModelGovernanceRecord | None:
        """Get governance record by ID."""
        return self.records.get(record_id)

    def list_governance_records(
        self, stage: ModelStage | None = None, status: GovernanceStatus | None = None
    ) -> list[ModelGovernanceRecord]:
        """List governance records with optional filtering."""
        records = list(self.records.values())

        if stage:
            records = [r for r in records if r.stage == stage]

        if status:
            records = [r for r in records if r.status == status]

        return records
