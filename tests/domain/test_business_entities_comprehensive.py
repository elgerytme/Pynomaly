"""Comprehensive tests for business-focused domain entities.

This module provides comprehensive test coverage for business-focused domain entities
including experiments, pipelines, alerts, governance, dashboards, and cost optimization.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pynomaly.domain.entities.alert import Alert
from pynomaly.domain.entities.cost_optimization import CostOptimization
from pynomaly.domain.entities.dashboard import Dashboard
from pynomaly.domain.entities.experiment import Experiment
from pynomaly.domain.entities.governance import Governance
from pynomaly.domain.entities.pipeline import Pipeline
from pynomaly.domain.exceptions import ValidationError


class TestExperiment:
    """Comprehensive tests for Experiment entity."""

    @pytest.fixture
    def sample_experiment(self):
        """Create sample experiment."""
        return Experiment(
            name="Algorithm Comparison Experiment",
            description="Compare IsolationForest vs LocalOutlierFactor performance",
            hypothesis="IsolationForest will show better precision on high-dimensional data",
            owner="data_scientist_1",
            tags=["comparison", "algorithm", "high_dimensional"],
        )

    def test_create_experiment(self, sample_experiment):
        """Test creating an experiment."""
        assert sample_experiment.name == "Algorithm Comparison Experiment"
        assert (
            sample_experiment.hypothesis
            == "IsolationForest will show better precision on high-dimensional data"
        )
        assert sample_experiment.owner == "data_scientist_1"
        assert sample_experiment.status == "draft"
        assert isinstance(sample_experiment.created_at, datetime)
        assert "comparison" in sample_experiment.tags

    def test_experiment_lifecycle(self, sample_experiment):
        """Test experiment lifecycle management."""
        # Start experiment
        sample_experiment.start()
        assert sample_experiment.status == "running"
        assert sample_experiment.started_at is not None

        # Add experimental runs
        run_1 = {
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1, "n_estimators": 100},
            "dataset_id": str(uuid4()),
            "results": {"precision": 0.92, "recall": 0.87, "f1_score": 0.89},
        }

        run_2 = {
            "algorithm": "LocalOutlierFactor",
            "parameters": {"contamination": 0.1, "n_neighbors": 20},
            "dataset_id": str(uuid4()),
            "results": {"precision": 0.88, "recall": 0.91, "f1_score": 0.89},
        }

        sample_experiment.add_run(run_1)
        sample_experiment.add_run(run_2)

        assert len(sample_experiment.runs) == 2
        assert sample_experiment.runs[0]["algorithm"] == "IsolationForest"

        # Complete experiment
        sample_experiment.complete()
        assert sample_experiment.status == "completed"
        assert sample_experiment.completed_at is not None

    def test_experiment_analysis(self, sample_experiment):
        """Test experiment analysis capabilities."""
        sample_experiment.start()

        # Add multiple runs with different configurations
        runs_data = [
            {
                "algorithm": "IsolationForest",
                "contamination": 0.05,
                "precision": 0.95,
                "recall": 0.82,
            },
            {
                "algorithm": "IsolationForest",
                "contamination": 0.10,
                "precision": 0.92,
                "recall": 0.87,
            },
            {
                "algorithm": "IsolationForest",
                "contamination": 0.15,
                "precision": 0.88,
                "recall": 0.91,
            },
            {
                "algorithm": "LocalOutlierFactor",
                "n_neighbors": 10,
                "precision": 0.89,
                "recall": 0.85,
            },
            {
                "algorithm": "LocalOutlierFactor",
                "n_neighbors": 20,
                "precision": 0.91,
                "recall": 0.88,
            },
            {
                "algorithm": "LocalOutlierFactor",
                "n_neighbors": 30,
                "precision": 0.87,
                "recall": 0.92,
            },
        ]

        for run_data in runs_data:
            sample_experiment.add_run(
                {
                    "algorithm": run_data["algorithm"],
                    "parameters": {
                        k: v
                        for k, v in run_data.items()
                        if k not in ["algorithm", "precision", "recall"]
                    },
                    "results": {
                        "precision": run_data["precision"],
                        "recall": run_data["recall"],
                    },
                }
            )

        # Analyze results
        analysis = sample_experiment.analyze_results()

        assert "best_run" in analysis
        assert "algorithm_comparison" in analysis
        assert "parameter_sensitivity" in analysis
        assert "statistical_significance" in analysis

        # Best run should have highest F1 score
        best_run = analysis["best_run"]
        assert "run_id" in best_run
        assert "performance_metrics" in best_run

    def test_experiment_validation(self, sample_experiment):
        """Test experiment validation."""
        # Valid experiment should pass validation
        assert sample_experiment.validate()

        # Invalid experiment - empty name
        invalid_experiment = Experiment(
            name="", description="Test", hypothesis="Test hypothesis", owner="test_user"
        )

        with pytest.raises(ValidationError):
            invalid_experiment.validate()

    def test_experiment_comparison(self, sample_experiment):
        """Test comparing multiple experiments."""
        # Create second experiment
        experiment_2 = Experiment(
            name="Parameter Tuning Experiment",
            description="Optimize IsolationForest parameters",
            hypothesis="Optimal contamination rate is around 0.05",
            owner="data_scientist_2",
        )

        # Add runs to both experiments
        sample_experiment.add_run(
            {
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1},
                "results": {"f1_score": 0.89},
            }
        )

        experiment_2.add_run(
            {
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.05},
                "results": {"f1_score": 0.92},
            }
        )

        # Compare experiments
        comparison = sample_experiment.compare_with(experiment_2)

        assert "performance_comparison" in comparison
        assert "methodology_differences" in comparison
        assert "statistical_tests" in comparison


class TestPipeline:
    """Comprehensive tests for Pipeline entity."""

    @pytest.fixture
    def sample_pipeline(self):
        """Create sample pipeline."""
        return Pipeline(
            name="Production Anomaly Detection Pipeline",
            description="End-to-end pipeline for production anomaly detection",
            owner="ml_engineer_1",
            schedule="0 */6 * * *",  # Every 6 hours
            pipeline_type="batch",
        )

    def test_create_pipeline(self, sample_pipeline):
        """Test creating a pipeline."""
        assert sample_pipeline.name == "Production Anomaly Detection Pipeline"
        assert sample_pipeline.pipeline_type == "batch"
        assert sample_pipeline.schedule == "0 */6 * * *"
        assert sample_pipeline.status == "inactive"
        assert isinstance(sample_pipeline.created_at, datetime)

    def test_pipeline_steps_configuration(self, sample_pipeline):
        """Test pipeline steps configuration."""
        # Define pipeline steps
        steps = [
            {
                "name": "data_ingestion",
                "type": "data_loader",
                "configuration": {
                    "source": "s3://data-bucket/anomaly-data/",
                    "format": "parquet",
                    "validation": True,
                },
                "retry_policy": {"max_retries": 3, "backoff_factor": 2},
            },
            {
                "name": "data_preprocessing",
                "type": "preprocessor",
                "configuration": {
                    "normalize": True,
                    "handle_missing": "impute",
                    "feature_selection": "variance_threshold",
                },
                "depends_on": ["data_ingestion"],
            },
            {
                "name": "anomaly_detection",
                "type": "detector",
                "configuration": {
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.1},
                    "model_version": "latest",
                },
                "depends_on": ["data_preprocessing"],
            },
            {
                "name": "result_storage",
                "type": "sink",
                "configuration": {
                    "destination": "postgresql://db:5432/anomalies",
                    "table": "detection_results",
                    "batch_size": 1000,
                },
                "depends_on": ["anomaly_detection"],
            },
        ]

        sample_pipeline.configure_steps(steps)

        assert len(sample_pipeline.steps) == 4
        assert sample_pipeline.steps[0]["name"] == "data_ingestion"
        assert sample_pipeline.steps[2]["depends_on"] == ["data_preprocessing"]

        # Validate step dependencies
        dependency_graph = sample_pipeline.get_dependency_graph()
        assert "data_ingestion" in dependency_graph
        assert dependency_graph["data_preprocessing"] == ["data_ingestion"]

    def test_pipeline_execution(self, sample_pipeline):
        """Test pipeline execution."""
        # Configure steps first
        steps = [
            {"name": "step1", "type": "loader", "configuration": {}},
            {
                "name": "step2",
                "type": "processor",
                "configuration": {},
                "depends_on": ["step1"],
            },
            {
                "name": "step3",
                "type": "detector",
                "configuration": {},
                "depends_on": ["step2"],
            },
        ]
        sample_pipeline.configure_steps(steps)

        # Start pipeline
        sample_pipeline.activate()
        assert sample_pipeline.status == "active"

        # Execute pipeline run
        run_id = sample_pipeline.execute()
        assert run_id is not None

        # Check execution status
        execution_status = sample_pipeline.get_execution_status(run_id)
        assert "run_id" in execution_status
        assert "status" in execution_status
        assert "started_at" in execution_status

        # Simulate step completion
        sample_pipeline.update_step_status(
            run_id, "step1", "completed", {"records_processed": 10000}
        )
        sample_pipeline.update_step_status(
            run_id, "step2", "completed", {"features_selected": 25}
        )
        sample_pipeline.update_step_status(
            run_id, "step3", "completed", {"anomalies_detected": 150}
        )

        # Mark run as completed
        sample_pipeline.complete_run(
            run_id, {"total_anomalies": 150, "execution_time_minutes": 15}
        )

        final_status = sample_pipeline.get_execution_status(run_id)
        assert final_status["status"] == "completed"

    def test_pipeline_monitoring(self, sample_pipeline):
        """Test pipeline monitoring capabilities."""
        # Configure monitoring settings
        monitoring_config = {
            "metrics": ["execution_time", "success_rate", "data_quality"],
            "alerting": {
                "failure_threshold": 2,  # Alert after 2 consecutive failures
                "performance_degradation_threshold": 0.3,  # 30% increase in execution time
                "data_quality_threshold": 0.95,
            },
            "retention_days": 30,
        }

        sample_pipeline.configure_monitoring(monitoring_config)

        # Record execution metrics
        metrics_data = [
            {
                "run_id": "run_1",
                "execution_time": 900,
                "success": True,
                "data_quality_score": 0.97,
            },
            {
                "run_id": "run_2",
                "execution_time": 950,
                "success": True,
                "data_quality_score": 0.96,
            },
            {
                "run_id": "run_3",
                "execution_time": 1200,
                "success": False,
                "error": "data_validation_failed",
            },
            {
                "run_id": "run_4",
                "execution_time": 1250,
                "success": False,
                "error": "connection_timeout",
            },
        ]

        for metrics in metrics_data:
            sample_pipeline.record_execution_metrics(metrics)

        # Get monitoring dashboard data
        dashboard_data = sample_pipeline.get_monitoring_dashboard()

        assert "success_rate" in dashboard_data
        assert "average_execution_time" in dashboard_data
        assert "recent_failures" in dashboard_data
        assert "performance_trend" in dashboard_data

        # Success rate should be 50% (2 success out of 4)
        assert abs(dashboard_data["success_rate"] - 0.5) < 0.01

    def test_pipeline_versioning(self, sample_pipeline):
        """Test pipeline versioning."""
        # Initial version
        initial_config = sample_pipeline.get_configuration()
        initial_version = sample_pipeline.create_version(
            "v1.0.0", "Initial pipeline version"
        )

        assert initial_version["version"] == "v1.0.0"
        assert initial_version["description"] == "Initial pipeline version"

        # Modify pipeline
        new_steps = [
            {"name": "enhanced_step", "type": "enhanced_processor", "configuration": {}}
        ]
        sample_pipeline.configure_steps(new_steps)

        # Create new version
        updated_version = sample_pipeline.create_version(
            "v1.1.0", "Added enhanced processing"
        )

        assert updated_version["version"] == "v1.1.0"

        # Rollback to previous version
        sample_pipeline.rollback_to_version("v1.0.0")

        current_config = sample_pipeline.get_configuration()
        assert (
            current_config != sample_pipeline.get_configuration()
        )  # Should be different from v1.1.0


class TestAlert:
    """Comprehensive tests for Alert entity."""

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert."""
        return Alert(
            name="High Anomaly Rate Alert",
            description="Triggers when anomaly rate exceeds threshold",
            alert_type="threshold",
            severity="high",
            created_by="ml_engineer_1",
        )

    def test_create_alert(self, sample_alert):
        """Test creating an alert."""
        assert sample_alert.name == "High Anomaly Rate Alert"
        assert sample_alert.alert_type == "threshold"
        assert sample_alert.severity == "high"
        assert sample_alert.status == "active"
        assert isinstance(sample_alert.created_at, datetime)

    def test_alert_conditions(self, sample_alert):
        """Test alert condition configuration."""
        # Configure threshold condition
        threshold_condition = {
            "metric": "anomaly_rate",
            "operator": ">",
            "threshold": 0.15,
            "evaluation_window": "5m",
            "consecutive_breaches": 2,
        }

        sample_alert.add_condition("threshold_breach", threshold_condition)

        # Configure trend condition
        trend_condition = {
            "metric": "anomaly_rate",
            "trend_type": "increasing",
            "trend_threshold": 0.05,
            "evaluation_window": "30m",
        }

        sample_alert.add_condition("trend_analysis", trend_condition)

        conditions = sample_alert.get_conditions()
        assert len(conditions) == 2
        assert conditions["threshold_breach"]["threshold"] == 0.15
        assert conditions["trend_analysis"]["trend_type"] == "increasing"

    def test_alert_evaluation(self, sample_alert):
        """Test alert condition evaluation."""
        # Configure simple threshold condition
        sample_alert.add_condition(
            "high_rate",
            {
                "metric": "anomaly_rate",
                "operator": ">",
                "threshold": 0.1,
                "evaluation_window": "5m",
            },
        )

        # Test data that should trigger alert
        test_data = [
            {"timestamp": datetime.now() - timedelta(minutes=4), "anomaly_rate": 0.12},
            {"timestamp": datetime.now() - timedelta(minutes=3), "anomaly_rate": 0.15},
            {"timestamp": datetime.now() - timedelta(minutes=2), "anomaly_rate": 0.18},
            {"timestamp": datetime.now() - timedelta(minutes=1), "anomaly_rate": 0.14},
        ]

        # Evaluate conditions
        evaluation_result = sample_alert.evaluate_conditions(test_data)

        assert evaluation_result["triggered"]
        assert "high_rate" in evaluation_result["triggered_conditions"]
        assert evaluation_result["max_severity"] == "high"

    def test_alert_notification(self, sample_alert):
        """Test alert notification system."""
        # Configure notification channels
        notification_config = {
            "email": {
                "recipients": ["team@company.com", "oncall@company.com"],
                "template": "anomaly_alert_template",
                "enabled": True,
            },
            "slack": {
                "channels": ["#ml-alerts", "#ops-alerts"],
                "webhook_url": "https://hooks.slack.com/webhook",
                "enabled": True,
            },
            "webhook": {
                "url": "https://api.company.com/alerts",
                "headers": {"Authorization": "Bearer token123"},
                "enabled": True,
            },
        }

        sample_alert.configure_notifications(notification_config)

        # Simulate alert trigger
        alert_data = {
            "triggered_at": datetime.now(),
            "metric_value": 0.18,
            "threshold": 0.1,
            "context": {"detector_id": str(uuid4()), "dataset_size": 10000},
        }

        # Send notifications
        notification_results = sample_alert.send_notifications(alert_data)

        assert "email" in notification_results
        assert "slack" in notification_results
        assert "webhook" in notification_results

        # All channels should be attempted
        for channel, result in notification_results.items():
            assert "status" in result
            assert "timestamp" in result

    def test_alert_escalation(self, sample_alert):
        """Test alert escalation policies."""
        # Configure escalation policy
        escalation_policy = {
            "levels": [
                {
                    "level": 1,
                    "delay_minutes": 0,
                    "recipients": ["primary-oncall@company.com"],
                    "channels": ["email"],
                },
                {
                    "level": 2,
                    "delay_minutes": 15,
                    "recipients": [
                        "secondary-oncall@company.com",
                        "manager@company.com",
                    ],
                    "channels": ["email", "slack"],
                },
                {
                    "level": 3,
                    "delay_minutes": 30,
                    "recipients": ["director@company.com"],
                    "channels": ["email", "phone"],
                },
            ],
            "max_escalations": 3,
            "escalation_interval": "15m",
        }

        sample_alert.configure_escalation(escalation_policy)

        # Simulate unacknowledged alert
        alert_instance = sample_alert.trigger_alert(
            {"metric_value": 0.25, "threshold": 0.1}
        )

        # Check initial escalation level
        assert alert_instance["escalation_level"] == 1

        # Simulate time passing without acknowledgment
        sample_alert.process_escalation(alert_instance["id"], minutes_elapsed=20)

        updated_instance = sample_alert.get_alert_instance(alert_instance["id"])
        assert updated_instance["escalation_level"] == 2

    def test_alert_suppression(self, sample_alert):
        """Test alert suppression and rate limiting."""
        # Configure suppression rules
        suppression_config = {
            "rate_limit": {"max_alerts_per_hour": 5, "max_alerts_per_day": 20},
            "suppression_window": "30m",
            "duplicate_suppression": True,
            "maintenance_mode_suppression": True,
        }

        sample_alert.configure_suppression(suppression_config)

        # Simulate multiple rapid alerts
        for i in range(10):
            alert_data = {"metric_value": 0.15 + i * 0.01, "threshold": 0.1}
            result = sample_alert.trigger_alert(alert_data)

            if i < 5:
                assert result["sent"]  # First 5 should be sent
            else:
                assert not result["sent"]  # Rest should be suppressed
                assert result["suppression_reason"] == "rate_limit_exceeded"


class TestGovernance:
    """Comprehensive tests for Governance entity."""

    @pytest.fixture
    def governance_framework(self):
        """Create governance framework."""
        return Governance(
            name="ML Model Governance Framework",
            description="Comprehensive governance for ML models",
            framework_version="2.1.0",
            compliance_standards=["SOX", "GDPR", "ISO27001"],
            created_by="governance_team",
        )

    def test_create_governance_framework(self, governance_framework):
        """Test creating governance framework."""
        assert governance_framework.name == "ML Model Governance Framework"
        assert governance_framework.framework_version == "2.1.0"
        assert "GDPR" in governance_framework.compliance_standards
        assert governance_framework.status == "draft"

    def test_governance_policies(self, governance_framework):
        """Test governance policy management."""
        # Define governance policies
        policies = [
            {
                "name": "Model Documentation Policy",
                "category": "documentation",
                "requirements": [
                    "model_card_required",
                    "performance_metrics_documented",
                    "bias_assessment_completed",
                    "data_lineage_tracked",
                ],
                "enforcement_level": "mandatory",
            },
            {
                "name": "Model Approval Policy",
                "category": "approval",
                "requirements": [
                    "peer_review_completed",
                    "stakeholder_signoff",
                    "risk_assessment_approved",
                ],
                "enforcement_level": "mandatory",
            },
            {
                "name": "Data Privacy Policy",
                "category": "privacy",
                "requirements": [
                    "pii_detection_completed",
                    "data_anonymization_verified",
                    "consent_management_implemented",
                ],
                "enforcement_level": "mandatory",
            },
        ]

        for policy in policies:
            governance_framework.add_policy(policy)

        assert len(governance_framework.get_policies()) == 3
        assert (
            governance_framework.get_policy("Model Documentation Policy")["category"]
            == "documentation"
        )

    def test_compliance_assessment(self, governance_framework):
        """Test compliance assessment."""
        # Add policies first
        governance_framework.add_policy(
            {
                "name": "Testing Policy",
                "requirements": [
                    "unit_tests",
                    "integration_tests",
                    "performance_tests",
                ],
                "enforcement_level": "mandatory",
            }
        )

        # Assess model compliance
        model_metadata = {
            "model_id": str(uuid4()),
            "name": "Production Anomaly Detector",
            "documentation": {
                "model_card": True,
                "performance_metrics": True,
                "bias_assessment": False,  # Missing
                "data_lineage": True,
            },
            "testing": {
                "unit_tests": True,
                "integration_tests": True,
                "performance_tests": False,  # Missing
            },
            "approval": {
                "peer_review": True,
                "stakeholder_signoff": False,  # Missing
                "risk_assessment": True,
            },
        }

        compliance_result = governance_framework.assess_compliance(model_metadata)

        assert "overall_compliance_score" in compliance_result
        assert "policy_compliance" in compliance_result
        assert "violations" in compliance_result
        assert "recommendations" in compliance_result

        # Should have violations for missing requirements
        violations = compliance_result["violations"]
        assert len(violations) > 0
        assert any("bias_assessment" in v["missing_requirement"] for v in violations)

    def test_governance_workflow(self, governance_framework):
        """Test governance workflow management."""
        # Define approval workflow
        workflow_config = {
            "stages": [
                {
                    "name": "development_review",
                    "approvers": ["lead_data_scientist"],
                    "requirements": ["code_review", "unit_tests"],
                    "auto_advance": False,
                },
                {
                    "name": "security_review",
                    "approvers": ["security_team"],
                    "requirements": ["security_scan", "vulnerability_assessment"],
                    "auto_advance": False,
                },
                {
                    "name": "business_approval",
                    "approvers": ["business_stakeholder", "product_manager"],
                    "requirements": ["business_validation", "impact_assessment"],
                    "auto_advance": False,
                },
                {
                    "name": "production_deployment",
                    "approvers": ["platform_team"],
                    "requirements": ["deployment_checklist", "monitoring_setup"],
                    "auto_advance": True,
                },
            ]
        }

        governance_framework.configure_workflow(workflow_config)

        # Start workflow for a model
        model_request = {
            "model_id": str(uuid4()),
            "model_name": "New Anomaly Detector",
            "requester": "data_scientist_1",
            "target_environment": "production",
        }

        workflow_instance = governance_framework.start_workflow(model_request)

        assert workflow_instance["status"] == "in_progress"
        assert workflow_instance["current_stage"] == "development_review"

        # Simulate stage approval
        approval_data = {
            "stage": "development_review",
            "approver": "lead_data_scientist",
            "decision": "approved",
            "comments": "Code review completed, all tests pass",
        }

        governance_framework.process_approval(workflow_instance["id"], approval_data)

        updated_instance = governance_framework.get_workflow_instance(
            workflow_instance["id"]
        )
        assert updated_instance["current_stage"] == "security_review"

    def test_audit_trail(self, governance_framework):
        """Test governance audit trail."""
        # Record governance events
        events = [
            {
                "event_type": "policy_created",
                "details": {"policy_name": "New Testing Policy"},
                "user": "governance_admin",
            },
            {
                "event_type": "compliance_assessment",
                "details": {"model_id": str(uuid4()), "score": 0.85},
                "user": "compliance_officer",
            },
            {
                "event_type": "workflow_started",
                "details": {"workflow_id": str(uuid4()), "model_name": "Test Model"},
                "user": "data_scientist_1",
            },
        ]

        for event in events:
            governance_framework.record_audit_event(
                event_type=event["event_type"],
                details=event["details"],
                user=event["user"],
            )

        # Generate audit report
        audit_report = governance_framework.generate_audit_report(
            start_date=datetime.now() - timedelta(days=30), end_date=datetime.now()
        )

        assert "total_events" in audit_report
        assert "events_by_type" in audit_report
        assert "user_activity" in audit_report
        assert "compliance_trends" in audit_report

        assert audit_report["total_events"] == 3


class TestDashboard:
    """Comprehensive tests for Dashboard entity."""

    @pytest.fixture
    def sample_dashboard(self):
        """Create sample dashboard."""
        return Dashboard(
            name="Anomaly Detection Operations Dashboard",
            description="Real-time monitoring of anomaly detection systems",
            dashboard_type="operational",
            owner="ops_team",
            refresh_interval=300,  # 5 minutes
        )

    def test_create_dashboard(self, sample_dashboard):
        """Test creating a dashboard."""
        assert sample_dashboard.name == "Anomaly Detection Operations Dashboard"
        assert sample_dashboard.dashboard_type == "operational"
        assert sample_dashboard.refresh_interval == 300
        assert sample_dashboard.is_public is False

    def test_dashboard_widgets(self, sample_dashboard):
        """Test dashboard widget configuration."""
        # Define various widget types
        widgets = [
            {
                "id": "anomaly_rate_metric",
                "type": "metric",
                "title": "Current Anomaly Rate",
                "data_source": "detection_results",
                "query": "SELECT AVG(anomaly_rate) FROM detection_results WHERE timestamp > NOW() - INTERVAL '1 hour'",
                "position": {"x": 0, "y": 0, "width": 2, "height": 1},
                "thresholds": {"warning": 0.1, "critical": 0.2},
            },
            {
                "id": "detection_volume_chart",
                "type": "time_series",
                "title": "Detection Volume Over Time",
                "data_source": "detection_results",
                "query": "SELECT timestamp, COUNT(*) FROM detection_results GROUP BY timestamp",
                "position": {"x": 2, "y": 0, "width": 4, "height": 2},
                "chart_config": {"line_color": "#1f77b4", "show_points": True},
            },
            {
                "id": "detector_status_table",
                "type": "table",
                "title": "Detector Status",
                "data_source": "detectors",
                "query": "SELECT name, status, last_trained, performance_score FROM detectors",
                "position": {"x": 0, "y": 1, "width": 6, "height": 3},
                "columns": ["name", "status", "last_trained", "performance_score"],
            },
            {
                "id": "anomaly_distribution",
                "type": "histogram",
                "title": "Anomaly Score Distribution",
                "data_source": "detection_results",
                "query": "SELECT anomaly_score FROM detection_results WHERE timestamp > NOW() - INTERVAL '24 hours'",
                "position": {"x": 6, "y": 0, "width": 3, "height": 2},
                "bins": 20,
            },
        ]

        for widget in widgets:
            sample_dashboard.add_widget(widget)

        assert len(sample_dashboard.get_widgets()) == 4
        assert sample_dashboard.get_widget("anomaly_rate_metric")["type"] == "metric"

    def test_dashboard_data_refresh(self, sample_dashboard):
        """Test dashboard data refresh."""
        # Add a widget first
        sample_dashboard.add_widget(
            {
                "id": "test_metric",
                "type": "metric",
                "data_source": "test_table",
                "query": "SELECT COUNT(*) as count FROM test_table",
            }
        )

        # Mock data refresh
        mock_data = {
            "test_metric": {
                "value": 1500,
                "timestamp": datetime.now(),
                "status": "success",
            }
        }

        sample_dashboard.refresh_data(mock_data)

        widget_data = sample_dashboard.get_widget_data("test_metric")
        assert widget_data["value"] == 1500
        assert widget_data["status"] == "success"

        # Check last refresh time
        assert sample_dashboard.last_refresh is not None

    def test_dashboard_sharing(self, sample_dashboard):
        """Test dashboard sharing and permissions."""
        # Configure sharing settings
        sharing_config = {
            "public": False,
            "shared_with_users": ["analyst_1", "manager_1"],
            "shared_with_groups": ["ops_team", "data_team"],
            "permissions": {
                "analyst_1": ["view"],
                "manager_1": ["view", "edit"],
                "ops_team": ["view"],
                "data_team": ["view", "edit"],
            },
        }

        sample_dashboard.configure_sharing(sharing_config)

        # Test permission checks
        assert sample_dashboard.can_view("analyst_1")
        assert sample_dashboard.can_edit("manager_1")
        assert not sample_dashboard.can_edit("analyst_1")
        assert sample_dashboard.can_view("ops_team_member")  # Group member

    def test_dashboard_alerts_integration(self, sample_dashboard):
        """Test dashboard integration with alerts."""
        # Configure dashboard alerts
        alert_configs = [
            {
                "widget_id": "anomaly_rate_metric",
                "condition": "value > 0.15",
                "severity": "warning",
                "message": "Anomaly rate exceeds warning threshold",
            },
            {
                "widget_id": "anomaly_rate_metric",
                "condition": "value > 0.25",
                "severity": "critical",
                "message": "Anomaly rate exceeds critical threshold",
            },
        ]

        for alert_config in alert_configs:
            sample_dashboard.add_alert(alert_config)

        # Simulate data update that triggers alert
        mock_data = {
            "anomaly_rate_metric": {
                "value": 0.18,  # Above warning threshold
                "timestamp": datetime.now(),
            }
        }

        sample_dashboard.refresh_data(mock_data)

        # Check for triggered alerts
        active_alerts = sample_dashboard.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0]["severity"] == "warning"


class TestCostOptimization:
    """Comprehensive tests for CostOptimization entity."""

    @pytest.fixture
    def cost_optimizer(self):
        """Create cost optimization instance."""
        return CostOptimization(
            name="ML Infrastructure Cost Optimizer",
            optimization_scope="infrastructure",
            target_cost_reduction=0.3,  # 30% reduction target
            created_by="finops_team",
        )

    def test_create_cost_optimizer(self, cost_optimizer):
        """Test creating cost optimization configuration."""
        assert cost_optimizer.name == "ML Infrastructure Cost Optimizer"
        assert cost_optimizer.optimization_scope == "infrastructure"
        assert cost_optimizer.target_cost_reduction == 0.3
        assert cost_optimizer.status == "inactive"

    def test_cost_analysis(self, cost_optimizer):
        """Test cost analysis capabilities."""
        # Mock cost data
        cost_data = {
            "compute_costs": {
                "training_instances": 2500.00,
                "inference_instances": 1800.00,
                "gpu_usage": 3200.00,
            },
            "storage_costs": {
                "model_storage": 150.00,
                "data_storage": 800.00,
                "backup_storage": 200.00,
            },
            "data_transfer_costs": {"ingress": 50.00, "egress": 300.00},
            "managed_services": {"ml_platform": 1200.00, "monitoring": 400.00},
        }

        cost_optimizer.set_current_costs(cost_data)

        # Analyze cost breakdown
        analysis = cost_optimizer.analyze_costs()

        assert "total_cost" in analysis
        assert "cost_breakdown" in analysis
        assert "cost_trends" in analysis
        assert "optimization_opportunities" in analysis

        total_cost = analysis["total_cost"]
        expected_total = 2500 + 1800 + 3200 + 150 + 800 + 200 + 50 + 300 + 1200 + 400
        assert abs(total_cost - expected_total) < 0.01

    def test_optimization_strategies(self, cost_optimizer):
        """Test cost optimization strategies."""
        # Define optimization strategies
        strategies = [
            {
                "name": "Right-size Instances",
                "category": "compute",
                "description": "Optimize instance types based on actual usage",
                "potential_savings": 0.25,
                "implementation_effort": "medium",
                "risk_level": "low",
            },
            {
                "name": "Spot Instance Usage",
                "category": "compute",
                "description": "Use spot instances for non-critical training workloads",
                "potential_savings": 0.70,
                "implementation_effort": "high",
                "risk_level": "medium",
            },
            {
                "name": "Data Lifecycle Management",
                "category": "storage",
                "description": "Implement automated data archiving and cleanup",
                "potential_savings": 0.40,
                "implementation_effort": "low",
                "risk_level": "low",
            },
            {
                "name": "Model Compression",
                "category": "inference",
                "description": "Compress models to reduce inference costs",
                "potential_savings": 0.30,
                "implementation_effort": "high",
                "risk_level": "medium",
            },
        ]

        for strategy in strategies:
            cost_optimizer.add_optimization_strategy(strategy)

        # Get recommended strategies
        recommendations = cost_optimizer.get_recommendations(
            max_risk_level="medium", min_potential_savings=0.2
        )

        assert len(recommendations) > 0
        assert all(rec["potential_savings"] >= 0.2 for rec in recommendations)
        assert all(rec["risk_level"] in ["low", "medium"] for rec in recommendations)

    def test_optimization_implementation(self, cost_optimizer):
        """Test optimization strategy implementation."""
        # Add a strategy
        strategy = {
            "name": "Automated Scaling",
            "category": "compute",
            "potential_savings": 0.35,
            "implementation_steps": [
                "Analyze usage patterns",
                "Configure auto-scaling policies",
                "Implement scaling triggers",
                "Monitor and adjust",
            ],
        }

        cost_optimizer.add_optimization_strategy(strategy)

        # Start implementation
        implementation = cost_optimizer.start_implementation("Automated Scaling")

        assert implementation["status"] == "in_progress"
        assert implementation["current_step"] == "Analyze usage patterns"

        # Complete implementation steps
        for i, step in enumerate(strategy["implementation_steps"]):
            cost_optimizer.complete_implementation_step(
                implementation["id"],
                step,
                {"completion_date": datetime.now(), "notes": f"Completed step {i + 1}"},
            )

        # Finalize implementation
        cost_optimizer.complete_implementation(
            implementation["id"], {"actual_savings": 0.32, "implementation_cost": 5000}
        )

        final_status = cost_optimizer.get_implementation_status(implementation["id"])
        assert final_status["status"] == "completed"
        assert final_status["actual_savings"] == 0.32

    def test_cost_monitoring(self, cost_optimizer):
        """Test ongoing cost monitoring."""
        # Set baseline costs
        baseline_costs = {
            "total": 10000,
            "compute": 7000,
            "storage": 2000,
            "other": 1000,
        }
        cost_optimizer.set_baseline_costs(baseline_costs)

        # Record cost data over time
        cost_timeline = [
            {"date": "2023-10-01", "total": 10000, "compute": 7000, "storage": 2000},
            {"date": "2023-10-02", "total": 9500, "compute": 6500, "storage": 2000},
            {"date": "2023-10-03", "total": 9200, "compute": 6200, "storage": 2000},
            {"date": "2023-10-04", "total": 8800, "compute": 5800, "storage": 2000},
        ]

        for cost_data in cost_timeline:
            cost_optimizer.record_daily_costs(cost_data)

        # Calculate savings progress
        savings_report = cost_optimizer.calculate_savings_progress()

        assert "total_savings" in savings_report
        assert "savings_percentage" in savings_report
        assert "target_progress" in savings_report

        # Should show 12% savings (10000 -> 8800)
        expected_savings_pct = (10000 - 8800) / 10000
        assert abs(savings_report["savings_percentage"] - expected_savings_pct) < 0.01
