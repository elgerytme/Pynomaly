"""Comprehensive tests for advanced domain entities.

This module provides comprehensive test coverage for advanced domain entities
including model versioning, deployment, continuous learning, drift detection,
A/B testing, explainable AI, security compliance, and governance.
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities.ab_testing import ABTesting
from monorepo.domain.entities.continuous_learning import ContinuousLearning
from monorepo.domain.entities.deployment import Deployment
from monorepo.domain.entities.drift_detection import DriftDetection
from monorepo.domain.entities.explainable_ai import ExplainableAI
from monorepo.domain.entities.model_registry import ModelRegistry
from monorepo.domain.entities.model_version import ModelVersion
from monorepo.domain.entities.security_compliance import SecurityCompliance
from monorepo.domain.exceptions import ValidationError
from monorepo.domain.value_objects import (
    ModelStorageInfo,
    PerformanceMetrics,
    SemanticVersion,
)


class TestModelVersion:
    """Comprehensive tests for ModelVersion entity."""

    @pytest.fixture
    def sample_model_version(self):
        """Create sample model version."""
        return ModelVersion(
            version=SemanticVersion("1.2.3"),
            model_id=uuid4(),
            created_by="data_scientist_1",
            storage_info=ModelStorageInfo(
                location="/models/isolation_forest_v1.2.3",
                size_bytes=1024000,
                checksum="abc123def456",
            ),
            performance_metrics=PerformanceMetrics(
                accuracy=0.95, precision=0.92, recall=0.88, f1_score=0.90
            ),
        )

    def test_create_model_version(self, sample_model_version):
        """Test creating a model version."""
        assert isinstance(sample_model_version.version, SemanticVersion)
        assert isinstance(sample_model_version.model_id, UUID)
        assert sample_model_version.created_by == "data_scientist_1"
        assert isinstance(sample_model_version.storage_info, ModelStorageInfo)
        assert isinstance(sample_model_version.performance_metrics, PerformanceMetrics)
        assert isinstance(sample_model_version.created_at, datetime)

    def test_model_version_comparison(self, sample_model_version):
        """Test model version comparison."""
        newer_version = ModelVersion(
            version=SemanticVersion("1.3.0"),
            model_id=sample_model_version.model_id,
            created_by="data_scientist_2",
        )

        assert newer_version > sample_model_version
        assert sample_model_version < newer_version
        assert newer_version != sample_model_version

    def test_model_version_validation(self):
        """Test model version validation."""
        with pytest.raises(ValidationError):
            ModelVersion(
                version=SemanticVersion("1.0.0"),
                model_id=uuid4(),
                created_by="",  # Empty creator
            )

    def test_model_version_metadata(self, sample_model_version):
        """Test model version metadata management."""
        # Add metadata
        sample_model_version.add_metadata("algorithm", "IsolationForest")
        sample_model_version.add_metadata("contamination", 0.1)
        sample_model_version.add_metadata("training_dataset", "production_data_v1")

        assert sample_model_version.get_metadata("algorithm") == "IsolationForest"
        assert sample_model_version.get_metadata("contamination") == 0.1

        # Update metadata
        sample_model_version.update_metadata("contamination", 0.05)
        assert sample_model_version.get_metadata("contamination") == 0.05

        # Remove metadata
        sample_model_version.remove_metadata("training_dataset")
        assert sample_model_version.get_metadata("training_dataset") is None


class TestModelRegistry:
    """Comprehensive tests for ModelRegistry entity."""

    @pytest.fixture
    def model_registry(self):
        """Create model registry."""
        return ModelRegistry(name="Production Registry")

    @pytest.fixture
    def sample_model_versions(self):
        """Create sample model versions."""
        model_id = uuid4()
        return [
            ModelVersion(
                version=SemanticVersion("1.0.0"),
                model_id=model_id,
                created_by="scientist_1",
            ),
            ModelVersion(
                version=SemanticVersion("1.1.0"),
                model_id=model_id,
                created_by="scientist_2",
            ),
            ModelVersion(
                version=SemanticVersion("2.0.0"),
                model_id=model_id,
                created_by="scientist_1",
            ),
        ]

    def test_register_model_version(self, model_registry, sample_model_versions):
        """Test registering model versions."""
        for version in sample_model_versions:
            model_registry.register_version(version)

        assert len(model_registry.get_all_versions()) == 3
        assert model_registry.get_latest_version() == sample_model_versions[-1]

    def test_model_registry_versioning(self, model_registry, sample_model_versions):
        """Test model registry versioning capabilities."""
        for version in sample_model_versions:
            model_registry.register_version(version)

        # Get specific version
        v1_1_0 = model_registry.get_version(SemanticVersion("1.1.0"))
        assert v1_1_0 == sample_model_versions[1]

        # Get versions by range
        major_v1_versions = model_registry.get_versions_by_range("1.x.x")
        assert len(major_v1_versions) == 2

        # Get versions by creator
        scientist1_versions = model_registry.get_versions_by_creator("scientist_1")
        assert len(scientist1_versions) == 2

    def test_model_promotion(self, model_registry, sample_model_versions):
        """Test model promotion through stages."""
        version = sample_model_versions[0]
        model_registry.register_version(version)

        # Promote through stages
        model_registry.promote_version(version.id, "development", "staging")
        assert model_registry.get_version_stage(version.id) == "staging"

        model_registry.promote_version(version.id, "staging", "production")
        assert model_registry.get_version_stage(version.id) == "production"

        # Test invalid promotion
        with pytest.raises(ValidationError):
            model_registry.promote_version(
                version.id, "staging", "production"
            )  # Already in production

    def test_model_deprecation(self, model_registry, sample_model_versions):
        """Test model version deprecation."""
        for version in sample_model_versions:
            model_registry.register_version(version)

        old_version = sample_model_versions[0]
        model_registry.deprecate_version(
            version_id=old_version.id,
            reason="Superseded by v2.0.0",
            replacement_version=sample_model_versions[-1].id,
        )

        assert model_registry.is_deprecated(old_version.id)
        deprecation_info = model_registry.get_deprecation_info(old_version.id)
        assert deprecation_info["reason"] == "Superseded by v2.0.0"
        assert deprecation_info["replacement_version"] == sample_model_versions[-1].id


class TestDeployment:
    """Comprehensive tests for Deployment entity."""

    @pytest.fixture
    def sample_deployment(self):
        """Create sample deployment."""
        return Deployment(
            name="Production Anomaly Detector",
            model_version_id=uuid4(),
            environment="production",
            target_platform="kubernetes",
            configuration={
                "replicas": 3,
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "autoscaling": True,
            },
        )

    def test_create_deployment(self, sample_deployment):
        """Test creating a deployment."""
        assert sample_deployment.name == "Production Anomaly Detector"
        assert sample_deployment.environment == "production"
        assert sample_deployment.target_platform == "kubernetes"
        assert sample_deployment.status == "created"
        assert isinstance(sample_deployment.created_at, datetime)

    def test_deployment_lifecycle(self, sample_deployment):
        """Test deployment lifecycle management."""
        # Deploy
        sample_deployment.deploy()
        assert sample_deployment.status == "deploying"
        assert sample_deployment.deployed_at is not None

        # Mark as ready
        sample_deployment.mark_ready()
        assert sample_deployment.status == "ready"

        # Scale deployment
        sample_deployment.scale(replicas=5)
        assert sample_deployment.configuration["replicas"] == 5

        # Stop deployment
        sample_deployment.stop()
        assert sample_deployment.status == "stopped"

        # Rollback
        sample_deployment.rollback(previous_version_id=uuid4())
        assert sample_deployment.status == "rolling_back"

    def test_deployment_health_monitoring(self, sample_deployment):
        """Test deployment health monitoring."""
        sample_deployment.deploy()
        sample_deployment.mark_ready()

        # Record health metrics
        health_metrics = {
            "cpu_usage": 0.6,
            "memory_usage": 0.8,
            "request_rate": 1000,
            "error_rate": 0.01,
            "response_time_p95": 150,
        }

        sample_deployment.record_health_metrics(health_metrics)

        latest_metrics = sample_deployment.get_latest_health_metrics()
        assert latest_metrics["cpu_usage"] == 0.6
        assert latest_metrics["error_rate"] == 0.01

        # Check health status
        assert sample_deployment.is_healthy()

        # Simulate unhealthy state
        unhealthy_metrics = {
            "cpu_usage": 0.95,
            "memory_usage": 0.98,
            "error_rate": 0.15,
        }

        sample_deployment.record_health_metrics(unhealthy_metrics)
        assert not sample_deployment.is_healthy()

    def test_deployment_configuration_management(self, sample_deployment):
        """Test deployment configuration management."""
        # Update configuration
        new_config = {"replicas": 5, "cpu_limit": "4000m", "memory_limit": "8Gi"}

        sample_deployment.update_configuration(new_config)
        assert sample_deployment.configuration["replicas"] == 5
        assert sample_deployment.configuration["cpu_limit"] == "4000m"

        # Validate configuration
        assert sample_deployment.validate_configuration()

        # Test invalid configuration
        invalid_config = {"replicas": -1}
        with pytest.raises(ValidationError):
            sample_deployment.update_configuration(invalid_config)


class TestContinuousLearning:
    """Comprehensive tests for ContinuousLearning entity."""

    @pytest.fixture
    def continuous_learning(self):
        """Create continuous learning instance."""
        return ContinuousLearning(
            name="Production Continuous Learning",
            model_id=uuid4(),
            learning_strategy="incremental",
            retrain_threshold=0.1,
            validation_strategy="holdout",
        )

    def test_create_continuous_learning(self, continuous_learning):
        """Test creating continuous learning configuration."""
        assert continuous_learning.name == "Production Continuous Learning"
        assert continuous_learning.learning_strategy == "incremental"
        assert continuous_learning.retrain_threshold == 0.1
        assert continuous_learning.is_active is False

    def test_learning_trigger_conditions(self, continuous_learning):
        """Test learning trigger conditions."""
        # Performance degradation trigger
        continuous_learning.add_trigger_condition(
            condition_type="performance_degradation", threshold=0.15, metric="f1_score"
        )

        # Data drift trigger
        continuous_learning.add_trigger_condition(
            condition_type="data_drift", threshold=0.3, detection_method="ks_test"
        )

        # Time-based trigger
        continuous_learning.add_trigger_condition(
            condition_type="scheduled", interval="weekly", day_of_week="sunday"
        )

        triggers = continuous_learning.get_trigger_conditions()
        assert len(triggers) == 3
        assert any(t["condition_type"] == "performance_degradation" for t in triggers)

    def test_learning_execution(self, continuous_learning):
        """Test learning execution workflow."""
        continuous_learning.activate()
        assert continuous_learning.is_active

        # Simulate learning trigger
        trigger_context = {
            "trigger_type": "performance_degradation",
            "current_performance": 0.82,
            "baseline_performance": 0.95,
            "degradation": 0.13,
        }

        learning_session = continuous_learning.trigger_learning(trigger_context)
        assert learning_session["status"] == "initiated"
        assert learning_session["trigger_reason"] == "performance_degradation"

        # Complete learning session
        learning_result = {
            "new_model_performance": 0.93,
            "improvement": 0.11,
            "training_samples": 10000,
            "validation_score": 0.91,
        }

        continuous_learning.complete_learning_session(
            session_id=learning_session["session_id"], result=learning_result
        )

        assert learning_session["status"] == "completed"

    def test_learning_strategies(self, continuous_learning):
        """Test different learning strategies."""
        # Test incremental learning
        incremental_config = continuous_learning.configure_incremental_learning(
            batch_size=1000, learning_rate_decay=0.95, memory_budget=1000000
        )

        assert incremental_config["batch_size"] == 1000
        assert incremental_config["learning_rate_decay"] == 0.95

        # Test periodic retraining
        continuous_learning.learning_strategy = "periodic_retrain"
        retrain_config = continuous_learning.configure_periodic_retraining(
            retrain_frequency="monthly",
            full_dataset_retrain=True,
            validation_holdout=0.2,
        )

        assert retrain_config["retrain_frequency"] == "monthly"
        assert retrain_config["full_dataset_retrain"] is True

    def test_model_selection_strategy(self, continuous_learning):
        """Test model selection in continuous learning."""
        # Configure champion-challenger strategy
        continuous_learning.configure_model_selection(
            strategy="champion_challenger",
            challenger_ratio=0.1,
            performance_window="7d",
            statistical_significance=0.05,
        )

        # Simulate model comparison
        champion_performance = {"accuracy": 0.92, "f1_score": 0.89}
        challenger_performance = {"accuracy": 0.94, "f1_score": 0.91}

        selection_result = continuous_learning.compare_models(
            champion_metrics=champion_performance,
            challenger_metrics=challenger_performance,
        )

        assert selection_result["recommendation"] == "promote_challenger"
        assert selection_result["confidence"] > 0.8


class TestDriftDetection:
    """Comprehensive tests for DriftDetection entity."""

    @pytest.fixture
    def drift_detection(self):
        """Create drift detection instance."""
        return DriftDetection(
            name="Production Drift Monitor",
            model_id=uuid4(),
            detection_methods=[
                "ks_test",
                "jensen_shannon",
                "population_stability_index",
            ],
            monitoring_window="24h",
            alert_threshold=0.1,
        )

    def test_create_drift_detection(self, drift_detection):
        """Test creating drift detection configuration."""
        assert drift_detection.name == "Production Drift Monitor"
        assert "ks_test" in drift_detection.detection_methods
        assert drift_detection.monitoring_window == "24h"
        assert drift_detection.alert_threshold == 0.1

    def test_data_drift_detection(self, drift_detection):
        """Test data drift detection methods."""
        # Reference data (training distribution)
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.exponential(1, 1000),
            }
        )

        # Current data (potential drift)
        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1.2, 500),  # Mean and variance shift
                "feature2": np.random.normal(0, 1, 500),  # No drift
                "feature3": np.random.exponential(1.5, 500),  # Scale shift
            }
        )

        drift_detection.set_reference_data(reference_data)
        drift_results = drift_detection.detect_drift(current_data)

        assert "overall_drift_score" in drift_results
        assert "feature_drift_scores" in drift_results
        assert "drift_detected" in drift_results
        assert "method_results" in drift_results

        # feature1 should show drift
        assert drift_results["feature_drift_scores"]["feature1"] > 0.1
        # feature2 should show minimal drift
        assert drift_results["feature_drift_scores"]["feature2"] < 0.1

    def test_concept_drift_detection(self, drift_detection):
        """Test concept drift detection."""
        # Historical predictions and outcomes
        historical_predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
        historical_outcomes = np.array([0, 0, 1, 1, 0])

        # Current predictions and outcomes
        current_predictions = np.array([0.7, 0.8, 0.2, 0.1, 0.9])
        current_outcomes = np.array([0, 0, 1, 1, 0])  # Different relationship

        concept_drift_result = drift_detection.detect_concept_drift(
            historical_predictions=historical_predictions,
            historical_outcomes=historical_outcomes,
            current_predictions=current_predictions,
            current_outcomes=current_outcomes,
        )

        assert "concept_drift_detected" in concept_drift_result
        assert "drift_magnitude" in concept_drift_result
        assert "confidence" in concept_drift_result

    def test_drift_alerting(self, drift_detection):
        """Test drift alerting system."""
        # Configure alert channels
        alert_config = {
            "email": ["data-team@company.com"],
            "slack": ["#ml-alerts"],
            "webhook": ["https://api.company.com/ml-alerts"],
        }

        drift_detection.configure_alerting(alert_config)

        # Simulate drift detection that triggers alert
        high_drift_result = {
            "overall_drift_score": 0.25,  # Above threshold
            "drift_detected": True,
            "critical_features": ["feature1", "feature3"],
        }

        alert_triggered = drift_detection.trigger_alert_if_needed(high_drift_result)
        assert alert_triggered

        # Verify alert details
        latest_alert = drift_detection.get_latest_alert()
        assert latest_alert["severity"] == "high"
        assert latest_alert["drift_score"] == 0.25


class TestABTesting:
    """Comprehensive tests for ABTesting entity."""

    @pytest.fixture
    def ab_testing(self):
        """Create A/B testing instance."""
        return ABTesting(
            name="Model Performance A/B Test",
            hypothesis="New model version performs better than current production model",
            control_model_id=uuid4(),
            treatment_model_id=uuid4(),
            traffic_split={"control": 0.7, "treatment": 0.3},
        )

    def test_create_ab_test(self, ab_testing):
        """Test creating A/B test."""
        assert ab_testing.name == "Model Performance A/B Test"
        assert ab_testing.traffic_split["control"] == 0.7
        assert ab_testing.traffic_split["treatment"] == 0.3
        assert ab_testing.status == "draft"

    def test_ab_test_configuration(self, ab_testing):
        """Test A/B test configuration."""
        # Configure success metrics
        success_metrics = [
            {"name": "precision", "target": 0.9, "weight": 0.4},
            {"name": "recall", "target": 0.85, "weight": 0.3},
            {"name": "f1_score", "target": 0.87, "weight": 0.3},
        ]

        ab_testing.configure_success_metrics(success_metrics)

        # Configure statistical parameters
        ab_testing.configure_statistical_parameters(
            significance_level=0.05,
            power=0.8,
            minimum_effect_size=0.02,
            sample_size_calculation="auto",
        )

        # Set test duration
        ab_testing.set_duration(days=14)

        assert len(ab_testing.success_metrics) == 3
        assert ab_testing.statistical_config["significance_level"] == 0.05
        assert ab_testing.planned_duration == timedelta(days=14)

    def test_ab_test_execution(self, ab_testing):
        """Test A/B test execution."""
        # Start test
        ab_testing.start_test()
        assert ab_testing.status == "running"
        assert ab_testing.started_at is not None

        # Record experiment results
        control_results = {
            "precision": [0.88, 0.89, 0.87, 0.90],
            "recall": [0.82, 0.84, 0.83, 0.85],
            "f1_score": [0.85, 0.86, 0.85, 0.87],
        }

        treatment_results = {
            "precision": [0.91, 0.92, 0.90, 0.93],
            "recall": [0.86, 0.87, 0.85, 0.88],
            "f1_score": [0.88, 0.89, 0.87, 0.90],
        }

        ab_testing.record_results("control", control_results)
        ab_testing.record_results("treatment", treatment_results)

        # Analyze results
        analysis_result = ab_testing.analyze_results()

        assert "statistical_significance" in analysis_result
        assert "effect_size" in analysis_result
        assert "confidence_intervals" in analysis_result
        assert "recommendation" in analysis_result

    def test_ab_test_stopping_rules(self, ab_testing):
        """Test A/B test stopping rules."""
        # Configure early stopping rules
        stopping_rules = [
            {"type": "futility", "threshold": 0.01},
            {"type": "superiority", "confidence": 0.99},
            {"type": "maximum_duration", "days": 30},
        ]

        ab_testing.configure_stopping_rules(stopping_rules)
        ab_testing.start_test()

        # Simulate strong positive results that trigger early stopping
        strong_treatment_results = {
            "precision": [0.95] * 100,
            "recall": [0.92] * 100,
            "f1_score": [0.93] * 100,
        }

        weak_control_results = {
            "precision": [0.80] * 100,
            "recall": [0.78] * 100,
            "f1_score": [0.79] * 100,
        }

        ab_testing.record_results("treatment", strong_treatment_results)
        ab_testing.record_results("control", weak_control_results)

        stopping_evaluation = ab_testing.evaluate_stopping_rules()
        assert stopping_evaluation["should_stop"]
        assert stopping_evaluation["reason"] == "superiority"


class TestExplainableAI:
    """Comprehensive tests for ExplainableAI entity."""

    @pytest.fixture
    def explainable_ai(self):
        """Create explainable AI instance."""
        return ExplainableAI(
            name="Production Explainability Service",
            model_id=uuid4(),
            explanation_methods=["shap", "lime", "feature_importance"],
            explanation_scope="global_and_local",
            update_frequency="daily",
        )

    def test_create_explainable_ai(self, explainable_ai):
        """Test creating explainable AI configuration."""
        assert explainable_ai.name == "Production Explainability Service"
        assert "shap" in explainable_ai.explanation_methods
        assert explainable_ai.explanation_scope == "global_and_local"
        assert explainable_ai.update_frequency == "daily"

    def test_global_explanations(self, explainable_ai):
        """Test global model explanations."""
        # Simulate model training data
        training_data = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 1000),
                "pressure": np.random.normal(1000, 50, 1000),
                "humidity": np.random.normal(60, 10, 1000),
            }
        )

        explainable_ai.set_training_data(training_data)

        # Generate global explanations
        global_explanation = explainable_ai.generate_global_explanation(
            method="feature_importance", include_interactions=True
        )

        assert "feature_importance" in global_explanation
        assert "feature_interactions" in global_explanation
        assert "model_behavior_summary" in global_explanation

        # All features should have importance scores
        assert "temperature" in global_explanation["feature_importance"]
        assert "pressure" in global_explanation["feature_importance"]
        assert "humidity" in global_explanation["feature_importance"]

    def test_local_explanations(self, explainable_ai):
        """Test local instance explanations."""
        # Single instance for explanation
        instance = {
            "temperature": 85,  # Anomalous value
            "pressure": 950,  # Low pressure
            "humidity": 30,  # Low humidity
        }

        local_explanation = explainable_ai.generate_local_explanation(
            instance=instance, method="shap", reference_data_size=1000
        )

        assert "feature_contributions" in local_explanation
        assert "base_value" in local_explanation
        assert "prediction_score" in local_explanation
        assert "explanation_text" in local_explanation

        # Temperature should have high contribution (anomalous)
        assert abs(local_explanation["feature_contributions"]["temperature"]) > 0.1

    def test_explanation_quality_monitoring(self, explainable_ai):
        """Test explanation quality monitoring."""
        # Generate multiple explanations
        instances = [
            {"temperature": 85, "pressure": 950, "humidity": 30},
            {"temperature": 15, "pressure": 1050, "humidity": 90},
            {"temperature": 100, "pressure": 800, "humidity": 10},
        ]

        explanations = []
        for instance in instances:
            explanation = explainable_ai.generate_local_explanation(
                instance=instance, method="lime"
            )
            explanations.append(explanation)

        # Monitor explanation quality
        quality_metrics = explainable_ai.monitor_explanation_quality(explanations)

        assert "consistency_score" in quality_metrics
        assert "completeness_score" in quality_metrics
        assert "stability_score" in quality_metrics
        assert "overall_quality" in quality_metrics

        assert 0.0 <= quality_metrics["overall_quality"] <= 1.0

    def test_explanation_bias_detection(self, explainable_ai):
        """Test bias detection in explanations."""
        # Create data with potential bias
        biased_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "protected_attr": np.random.choice(
                    [0, 1], 1000
                ),  # Binary protected attribute
            }
        )

        # Generate explanations for different groups
        group_0_explanations = []
        group_1_explanations = []

        for i in range(100):
            if biased_data.iloc[i]["protected_attr"] == 0:
                explanation = explainable_ai.generate_local_explanation(
                    instance=biased_data.iloc[i].to_dict(), method="shap"
                )
                group_0_explanations.append(explanation)
            else:
                explanation = explainable_ai.generate_local_explanation(
                    instance=biased_data.iloc[i].to_dict(), method="shap"
                )
                group_1_explanations.append(explanation)

        # Detect bias in explanations
        bias_analysis = explainable_ai.detect_explanation_bias(
            group_0_explanations=group_0_explanations,
            group_1_explanations=group_1_explanations,
            protected_attribute="protected_attr",
        )

        assert "bias_score" in bias_analysis
        assert "feature_bias_scores" in bias_analysis
        assert "fairness_metrics" in bias_analysis
        assert "bias_detected" in bias_analysis


class TestSecurityCompliance:
    """Comprehensive tests for SecurityCompliance entity."""

    @pytest.fixture
    def security_compliance(self):
        """Create security compliance instance."""
        return SecurityCompliance(
            name="GDPR ML Compliance",
            compliance_frameworks=["GDPR", "CCPA", "SOX"],
            security_level="high",
            audit_frequency="monthly",
        )

    def test_create_security_compliance(self, security_compliance):
        """Test creating security compliance configuration."""
        assert security_compliance.name == "GDPR ML Compliance"
        assert "GDPR" in security_compliance.compliance_frameworks
        assert security_compliance.security_level == "high"
        assert security_compliance.audit_frequency == "monthly"

    def test_data_protection_compliance(self, security_compliance):
        """Test data protection compliance checks."""
        # Configure data protection requirements
        data_protection_config = {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "data_anonymization": True,
            "access_logging": True,
            "retention_policy": "2_years",
            "right_to_be_forgotten": True,
        }

        security_compliance.configure_data_protection(data_protection_config)

        # Perform compliance check
        compliance_result = security_compliance.check_data_protection_compliance()

        assert "encryption_compliance" in compliance_result
        assert "anonymization_compliance" in compliance_result
        assert "access_control_compliance" in compliance_result
        assert "overall_compliance_score" in compliance_result

        assert compliance_result["overall_compliance_score"] >= 0.8

    def test_model_governance_compliance(self, security_compliance):
        """Test model governance compliance."""
        # Configure model governance requirements
        governance_config = {
            "model_documentation": True,
            "bias_testing": True,
            "performance_monitoring": True,
            "version_control": True,
            "approval_workflow": True,
            "explainability_requirements": True,
        }

        security_compliance.configure_model_governance(governance_config)

        # Mock model metadata for compliance check
        model_metadata = {
            "documentation_complete": True,
            "bias_tests_passed": True,
            "performance_baselines": True,
            "version_tracked": True,
            "approval_status": "approved",
            "explainability_enabled": True,
        }

        governance_result = security_compliance.check_model_governance(model_metadata)

        assert "documentation_compliance" in governance_result
        assert "bias_compliance" in governance_result
        assert "monitoring_compliance" in governance_result
        assert governance_result["overall_governance_score"] >= 0.9

    def test_audit_trail_management(self, security_compliance):
        """Test audit trail management."""
        # Log various events
        events = [
            {
                "event_type": "model_training",
                "user": "data_scientist_1",
                "timestamp": datetime.now(),
            },
            {
                "event_type": "model_deployment",
                "user": "ml_engineer_1",
                "timestamp": datetime.now(),
            },
            {
                "event_type": "data_access",
                "user": "analyst_1",
                "timestamp": datetime.now(),
            },
            {
                "event_type": "prediction_request",
                "user": "api_client",
                "timestamp": datetime.now(),
            },
        ]

        for event in events:
            security_compliance.log_audit_event(
                event_type=event["event_type"],
                user=event["user"],
                details={"action": f"Performed {event['event_type']}"},
                timestamp=event["timestamp"],
            )

        # Generate audit report
        audit_report = security_compliance.generate_audit_report(
            start_date=datetime.now() - timedelta(days=30), end_date=datetime.now()
        )

        assert "total_events" in audit_report
        assert "events_by_type" in audit_report
        assert "users_activity" in audit_report
        assert "compliance_violations" in audit_report

        assert audit_report["total_events"] == 4

    def test_privacy_compliance(self, security_compliance):
        """Test privacy compliance features."""
        # Configure privacy settings
        privacy_config = {
            "data_minimization": True,
            "purpose_limitation": True,
            "consent_management": True,
            "data_subject_rights": True,
        }

        security_compliance.configure_privacy_compliance(privacy_config)

        # Test data subject rights
        data_subject_request = {
            "subject_id": "user_123",
            "request_type": "data_access",
            "verification_status": "verified",
        }

        response = security_compliance.handle_data_subject_request(data_subject_request)
        assert response["status"] == "approved"
        assert "data_summary" in response

        # Test data deletion request
        deletion_request = {
            "subject_id": "user_123",
            "request_type": "data_deletion",
            "verification_status": "verified",
        }

        deletion_response = security_compliance.handle_data_subject_request(
            deletion_request
        )
        assert deletion_response["status"] == "approved"
        assert "deletion_confirmation" in deletion_response
