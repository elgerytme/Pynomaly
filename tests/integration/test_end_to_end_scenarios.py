"""End-to-end integration tests for complete user scenarios."""

import asyncio
import tempfile

import pytest
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestEndToEndScenarios:
    """Test complete end-to-end user scenarios and workflows."""

    @pytest.mark.asyncio
    async def test_data_scientist_research_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        sample_time_series_csv: str,
        disable_auth,
    ):
        """Test complete data scientist research and experimentation workflow."""

        # Scenario: A data scientist wants to research the best anomaly detection
        # algorithm for their specific dataset through systematic experimentation

        # Step 1: Data Upload and Exploration
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "research_dataset"
        )

        # Explore dataset characteristics
        response = await async_test_client.get(f"/api/datasets/{dataset['id']}/stats")
        response.raise_for_status()
        stats = response.json()["data"]

        assert "column_stats" in stats
        assert "data_quality" in stats

        # Validate data quality
        response = await async_test_client.post(
            f"/api/datasets/{dataset['id']}/validate"
        )
        response.raise_for_status()
        validation = response.json()["data"]
        assert validation["is_valid"] is True

        # Step 2: Create Multiple Detectors for Comparison
        algorithms_to_test = [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope",
        ]

        detectors = []
        for algorithm in algorithms_to_test:
            detector_config = {
                "name": f"research_{algorithm}_detector",
                "description": f"Research detector using {algorithm}",
                "algorithm": algorithm,
                "parameters": self._get_algorithm_parameters(algorithm),
                "feature_columns": ["feature1", "feature2", "feature3"],
            }

            response = await async_test_client.post(
                f"/api/detectors/create?dataset_id={dataset['id']}",
                json=detector_config,
            )
            response.raise_for_status()
            detector = response.json()["data"]
            detectors.append(detector)
            integration_helper.created_resources["detectors"].append(detector["id"])

        # Step 3: Train All Detectors
        training_results = []
        for detector in detectors:
            response = await async_test_client.post(
                f"/api/detection/train/{detector['id']}"
            )
            response.raise_for_status()
            result = response.json()["data"]
            training_results.append(result)

        # Verify all training completed successfully
        for result in training_results:
            assert result["status"] == "completed"
            assert "model_id" in result
            assert "metrics" in result

        # Step 4: Comprehensive Algorithm Comparison Experiment
        experiment_config = {
            "name": "algorithm_comparison_research",
            "description": "Systematic comparison of anomaly detection algorithms",
            "dataset_id": dataset["id"],
            "algorithms": [
                {"name": algo, "parameters": self._get_algorithm_parameters(algo)}
                for algo in algorithms_to_test
            ],
            "evaluation_metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
                "matthews_corrcoef",
            ],
            "cross_validation": {
                "enabled": True,
                "folds": 5,
                "stratified": True,
                "random_state": 42,
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "strategy": "grid_search",
                "n_trials": 10,
            },
        }

        response = await async_test_client.post(
            "/api/experiments/create?created_by=data_scientist", json=experiment_config
        )
        response.raise_for_status()
        experiment = response.json()["data"]
        integration_helper.created_resources["experiments"].append(experiment["id"])

        # Run experiment
        response = await async_test_client.post(
            f"/api/experiments/{experiment['id']}/run"
        )
        response.raise_for_status()

        # Step 5: Analyze Experiment Results
        response = await async_test_client.get(
            f"/api/experiments/{experiment['id']}/results"
        )
        response.raise_for_status()
        results = response.json()["data"]

        assert len(results["algorithm_results"]) == 4

        # Get detailed comparison
        response = await async_test_client.get(
            f"/api/experiments/{experiment['id']}/comparison"
        )
        response.raise_for_status()
        comparison = response.json()["data"]

        best_algorithm = comparison["best_algorithm"]
        assert best_algorithm["algorithm_name"] in algorithms_to_test

        # Step 6: Hyperparameter Optimization for Best Algorithm
        automl_config = {
            "job_name": "hyperparameter_optimization",
            "dataset_id": dataset["id"],
            "algorithms": [best_algorithm["algorithm_name"]],
            "optimization_metric": "f1_score",
            "max_trials": 50,
            "max_duration_minutes": 30,
            "hyperparameter_tuning": {
                "enabled": True,
                "strategy": "bayesian",
                "n_trials": 30,
            },
        }

        response = await async_test_client.post(
            "/api/autonomous/automl/start?created_by=data_scientist", json=automl_config
        )
        response.raise_for_status()
        automl_job = response.json()["data"]

        # Step 7: Model Performance Analysis
        best_model_id = best_algorithm["model_id"]

        # Performance benchmarking
        benchmark_config = {
            "model_id": best_model_id,
            "test_dataset_id": dataset["id"],
            "metrics": ["prediction_latency", "throughput", "memory_usage"],
            "test_scenarios": [
                {"name": "single_prediction", "batch_size": 1, "iterations": 100},
                {"name": "batch_prediction", "batch_size": 100, "iterations": 10},
            ],
        }

        response = await async_test_client.post(
            "/api/performance/benchmark", json=benchmark_config
        )
        response.raise_for_status()

        # Model explainability analysis
        response = await async_test_client.post(
            f"/api/models/{best_model_id}/explain",
            json={"explanation_type": "global", "sample_size": 100},
        )
        response.raise_for_status()
        explanation = response.json()["data"]

        assert "feature_importance" in explanation
        assert "explanation_plots" in explanation

        # Step 8: Export Research Results
        export_config = {
            "format": "research_report",
            "include_models": True,
            "include_data": False,
            "include_visualizations": True,
        }

        response = await async_test_client.post(
            f"/api/experiments/{experiment['id']}/export", json=export_config
        )
        response.raise_for_status()
        export_result = response.json()["data"]

        assert "download_url" in export_result
        assert export_result["format"] == "research_report"

    @pytest.mark.asyncio
    async def test_production_deployment_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test complete production deployment workflow."""

        # Scenario: An ML engineer wants to deploy a trained model to production
        # with proper validation, monitoring, and rollback capabilities

        # Step 1: Model Development and Validation
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "production_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        training_result = await integration_helper.train_detector(detector["id"])
        model_id = training_result["model_id"]

        # Comprehensive model validation
        response = await async_test_client.post(
            f"/api/detection/validate/{detector['id']}"
        )
        response.raise_for_status()
        validation = response.json()["data"]

        assert validation["metrics"]["accuracy"] > 0.8  # Production threshold
        assert validation["status"] == "completed"

        # Step 2: Model Registration and Staging
        registration_config = {
            "name": "production_anomaly_detector",
            "version": "1.0.0",
            "description": "Production-ready anomaly detection model",
            "stage": "staging",
            "tags": ["production", "validated", "v1"],
            "metadata": {
                "accuracy": validation["metrics"]["accuracy"],
                "precision": validation["metrics"]["precision"],
                "recall": validation["metrics"]["recall"],
                "validation_dataset": dataset["id"],
                "approved_by": "ml_engineer",
            },
        }

        response = await async_test_client.post(
            f"/api/models/{model_id}/register", json=registration_config
        )
        response.raise_for_status()
        registered_model = response.json()["data"]

        assert registered_model["stage"] == "staging"
        assert registered_model["name"] == "production_anomaly_detector"

        # Step 3: A/B Testing Setup
        # Create second model for comparison
        detector_v2 = await integration_helper.create_detector(
            dataset["id"], "one_class_svm"
        )

        training_result_v2 = await integration_helper.train_detector(detector_v2["id"])
        model_v2_id = training_result_v2["model_id"]

        # A/B test configuration
        ab_test_config = {
            "test_name": "production_model_comparison",
            "control_model_id": model_id,
            "treatment_model_id": model_v2_id,
            "traffic_split": {"control": 0.7, "treatment": 0.3},
            "success_metrics": ["accuracy", "latency", "throughput"],
            "test_duration_days": 7,
            "statistical_power": 0.8,
        }

        response = await async_test_client.post(
            "/api/models/ab-test/create", json=ab_test_config
        )
        response.raise_for_status()
        ab_test = response.json()["data"]

        # Step 4: Production Deployment with Monitoring
        deployment_config = {
            "model_id": model_id,
            "deployment_target": "production",
            "scaling_config": {
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70,
            },
            "monitoring_config": {
                "enable_drift_detection": True,
                "drift_threshold": 0.1,
                "performance_monitoring": True,
                "alert_thresholds": {
                    "latency_p99": 100,  # ms
                    "error_rate": 0.01,  # 1%
                    "throughput_min": 100,  # requests/sec
                },
            },
            "rollback_config": {
                "auto_rollback": True,
                "rollback_threshold": {"error_rate": 0.05, "latency_increase": 2.0},
            },
        }

        response = await async_test_client.post(
            f"/api/models/{model_id}/deploy", json=deployment_config
        )
        response.raise_for_status()
        deployment = response.json()["data"]

        assert deployment["status"] in ["deploying", "deployed"]
        assert deployment["deployment_target"] == "production"

        # Step 5: Production Health Checks
        response = await async_test_client.get(f"/api/models/{model_id}/health")
        response.raise_for_status()
        health = response.json()["data"]

        assert health["status"] in ["healthy", "deploying"]
        assert "performance_metrics" in health
        assert "resource_usage" in health

        # Step 6: Live Traffic Testing
        # Simulate production predictions
        production_test_data = [
            {"feature1": 0.1, "feature2": 0.2, "feature3": 0.3},
            {"feature1": 0.5, "feature2": 0.4, "feature3": 0.6},
            {"feature1": 5.0, "feature2": 4.8, "feature3": 3.2},  # Anomaly
        ]

        prediction_results = []
        for test_data in production_test_data:
            response = await async_test_client.post(
                f"/api/detection/predict/{detector['id']}", json={"data": [test_data]}
            )
            response.raise_for_status()
            result = response.json()["data"]
            prediction_results.append(result[0])

        # Verify predictions
        assert len(prediction_results) == 3
        assert prediction_results[2]["is_anomaly"] is True

        # Step 7: Model Monitoring and Drift Detection
        response = await async_test_client.get(
            f"/api/models/{model_id}/monitoring/drift"
        )
        response.raise_for_status()
        drift_analysis = response.json()["data"]

        assert "data_drift" in drift_analysis
        assert "concept_drift" in drift_analysis
        assert "drift_score" in drift_analysis

        # Performance monitoring
        response = await async_test_client.get(
            f"/api/models/{model_id}/monitoring/performance"
        )
        response.raise_for_status()
        performance = response.json()["data"]

        assert "latency_metrics" in performance
        assert "throughput_metrics" in performance
        assert "error_metrics" in performance

        # Step 8: Model Promotion to Production
        promotion_config = {
            "stage": "production",
            "approved_by": "ml_engineer",
            "approval_notes": "Model passed all validation and monitoring checks",
            "deployment_strategy": "blue_green",
        }

        response = await async_test_client.post(
            f"/api/models/{model_id}/promote", json=promotion_config
        )
        response.raise_for_status()
        promoted_model = response.json()["data"]

        assert promoted_model["stage"] == "production"

        # Step 9: Automated Alerts and Notifications
        alert_config = {
            "alert_name": "production_model_performance",
            "model_id": model_id,
            "alert_conditions": [
                {
                    "metric": "error_rate",
                    "threshold": 0.02,
                    "comparison": "greater_than",
                    "duration_minutes": 5,
                },
                {
                    "metric": "latency_p99",
                    "threshold": 150,
                    "comparison": "greater_than",
                    "duration_minutes": 3,
                },
            ],
            "notification_channels": ["email", "slack", "webhook"],
            "escalation_policy": {
                "escalate_after_minutes": 15,
                "escalation_contacts": ["ml_engineer", "devops_team"],
            },
        }

        response = await async_test_client.post(
            "/api/models/alerts/create", json=alert_config
        )
        response.raise_for_status()
        alert = response.json()["data"]

        assert alert["alert_name"] == "production_model_performance"
        assert len(alert["alert_conditions"]) == 2

    @pytest.mark.asyncio
    async def test_security_analyst_monitoring_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_time_series_csv: str,
        disable_auth,
    ):
        """Test security analyst real-time monitoring and incident response workflow."""

        # Scenario: A security analyst needs to set up real-time monitoring
        # for network anomalies and respond to security incidents

        # Step 1: Security Dataset Setup
        dataset = await integration_helper.upload_dataset(
            sample_time_series_csv, "security_monitoring_dataset"
        )

        # Step 2: Security-Focused Detector Configuration
        security_detector_config = {
            "name": "network_security_monitor",
            "description": "Real-time network anomaly detection for security monitoring",
            "algorithm": "time_series_anomaly",
            "parameters": {
                "contamination": 0.02,  # Lower threshold for security
                "seasonal_period": 288,  # Daily pattern (5-min intervals)
                "trend_detection": True,
                "anomaly_sensitivity": "high",
            },
            "feature_columns": ["value", "cpu_usage", "memory_usage", "network_io"],
        }

        response = await async_test_client.post(
            f"/api/detectors/create?dataset_id={dataset['id']}",
            json=security_detector_config,
        )
        response.raise_for_status()
        detector = response.json()["data"]
        integration_helper.created_resources["detectors"].append(detector["id"])

        # Train security model
        await integration_helper.train_detector(detector["id"])

        # Step 3: Real-Time Streaming Security Monitor
        security_stream_config = {
            "name": "security_monitoring_stream",
            "description": "24/7 security monitoring stream",
            "detector_id": detector["id"],
            "data_source": {
                "source_type": "kafka",
                "connection_config": {
                    "bootstrap_servers": ["security-kafka:9092"],
                    "topic": "network_events",
                    "security_protocol": "SSL",
                },
            },
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 1000,
                "schema_validation": True,
                "enable_checkpointing": True,
                "checkpoint_interval_seconds": 10,
            },
            "data_sink": {
                "sink_type": "security_siem",
                "connection_config": {
                    "endpoint": "https://siem.company.com/api/events",
                    "api_key_env": "SIEM_API_KEY",
                },
            },
            "tags": ["security", "critical", "24x7"],
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=security_analyst",
            json=security_stream_config,
        )
        response.raise_for_status()
        security_session = response.json()["data"]
        integration_helper.created_resources["sessions"].append(security_session["id"])

        # Start security monitoring
        response = await async_test_client.post(
            f"/api/streaming/sessions/{security_session['id']}/start"
        )
        response.raise_for_status()

        # Step 4: Security Alert Patterns Configuration
        security_patterns = [
            {
                "name": "Potential DDoS Attack",
                "pattern_type": "frequency",
                "conditions": {
                    "event_type": "anomaly_detected",
                    "min_count": 10,
                    "severity": "high",
                    "time_window_seconds": 300,
                },
                "description": "Multiple high-severity anomalies indicating potential DDoS",
                "confidence": 0.9,
                "alert_threshold": 1,
            },
            {
                "name": "Data Exfiltration Pattern",
                "pattern_type": "sequence",
                "conditions": {
                    "sequence": [
                        {
                            "event_type": "anomaly_detected",
                            "feature": "network_io",
                            "threshold": 1000000,
                        },
                        {
                            "event_type": "anomaly_detected",
                            "feature": "cpu_usage",
                            "threshold": 80,
                        },
                        {
                            "event_type": "anomaly_detected",
                            "feature": "memory_usage",
                            "threshold": 85,
                        },
                    ],
                    "time_window_seconds": 600,
                },
                "description": "Sequence pattern indicating potential data exfiltration",
                "confidence": 0.85,
                "alert_threshold": 1,
            },
        ]

        created_patterns = []
        for pattern_config in security_patterns:
            response = await async_test_client.post(
                "/api/events/patterns?created_by=security_analyst", json=pattern_config
            )
            response.raise_for_status()
            pattern = response.json()["data"]
            created_patterns.append(pattern)

        # Step 5: High-Priority Security Alerts
        security_alerts = [
            {
                "name": "Critical Anomaly Rate Alert",
                "metric_name": "anomaly_rate",
                "threshold_value": 0.05,  # 5% anomaly rate
                "comparison_operator": ">",
                "severity": "critical",
                "duration_threshold_minutes": 2.0,
                "notification_channels": ["sms", "email", "slack", "pagerduty"],
            },
            {
                "name": "Processing Lag Alert",
                "metric_name": "avg_processing_time",
                "threshold_value": 1000,  # 1 second processing time
                "comparison_operator": ">",
                "severity": "high",
                "duration_threshold_minutes": 5.0,
                "notification_channels": ["email", "slack"],
            },
        ]

        for alert_config in security_alerts:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{security_session['id']}/alerts?created_by=security_analyst",
                json=alert_config,
            )
            response.raise_for_status()

        # Step 6: Simulate Security Events
        security_events = [
            # Normal network activity
            {
                "timestamp": "2024-12-25T10:00:00Z",
                "value": 50.0,
                "cpu_usage": 25.0,
                "memory_usage": 40.0,
                "network_io": 1024,
            },
            {
                "timestamp": "2024-12-25T10:01:00Z",
                "value": 52.0,
                "cpu_usage": 27.0,
                "memory_usage": 42.0,
                "network_io": 1100,
            },
            # Suspicious activity - potential attack
            {
                "timestamp": "2024-12-25T10:02:00Z",
                "value": 200.0,
                "cpu_usage": 85.0,
                "memory_usage": 88.0,
                "network_io": 10240000,
            },
            {
                "timestamp": "2024-12-25T10:03:00Z",
                "value": 250.0,
                "cpu_usage": 92.0,
                "memory_usage": 95.0,
                "network_io": 15360000,
            },
            {
                "timestamp": "2024-12-25T10:04:00Z",
                "value": 180.0,
                "cpu_usage": 87.0,
                "memory_usage": 90.0,
                "network_io": 8192000,
            },
        ]

        incident_events = []
        for event_data in security_events:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{security_session['id']}/process",
                json={"data": event_data},
            )
            response.raise_for_status()
            result = response.json()["data"]
            incident_events.append(result)

        # Step 7: Security Incident Response
        # Query high-severity anomaly events
        incident_query = {
            "event_types": ["anomaly_detected"],
            "severities": ["high", "critical"],
            "min_anomaly_score": 0.7,
            "event_time_start": "2024-12-25T10:00:00Z",
            "event_time_end": "2024-12-25T10:10:00Z",
            "limit": 50,
        }

        response = await async_test_client.post(
            "/api/events/query", json=incident_query
        )
        response.raise_for_status()
        security_incidents = response.json()["data"]

        # Step 8: Incident Triage and Response
        for incident in security_incidents:
            if incident["severity"] in ["high", "critical"]:
                # Acknowledge incident
                response = await async_test_client.post(
                    f"/api/events/{incident['id']}/acknowledge?user=security_analyst",
                    json={"notes": "Security incident under investigation"},
                )
                response.raise_for_status()

                # Create incident investigation case
                investigation_data = {
                    "incident_id": incident["id"],
                    "severity": incident["severity"],
                    "summary": f"Anomaly detected: {incident['title']}",
                    "assigned_to": "security_analyst",
                    "status": "investigating",
                }

                # In a real system, this would create a case in the security system
                # For testing, we'll verify the incident data
                assert incident["anomaly_data"]["anomaly_score"] > 0.7
                assert "feature_contributions" in incident["anomaly_data"]

        # Step 9: Real-Time Dashboard Monitoring
        response = await async_test_client.get(
            f"/api/streaming/sessions/{security_session['id']}/metrics"
        )
        response.raise_for_status()
        real_time_metrics = response.json()["data"]

        assert real_time_metrics["messages_processed"] >= 5
        assert (
            real_time_metrics["anomalies_detected"] >= 3
        )  # Should detect the suspicious events

        # Step 10: Threat Intelligence Integration
        threat_intel_query = {
            "indicators": ["high_network_io", "high_cpu_usage", "high_memory_usage"],
            "time_range": "1_hour",
            "correlation_threshold": 0.8,
        }

        # Simulate threat intelligence correlation
        # In a real system, this would integrate with threat intelligence feeds
        response = await async_test_client.post(
            "/api/security/threat-intel/correlate", json=threat_intel_query
        )
        # This endpoint might not be implemented yet, so handle gracefully
        if response.status_code == 404:
            # Expected if endpoint not implemented
            pass
        else:
            response.raise_for_status()

        # Step 11: Incident Reporting and Closure
        incident_report = {
            "incident_summary": "Multiple high-severity network anomalies detected",
            "affected_systems": ["network_monitor", "security_stream"],
            "timeline": security_events,
            "root_cause": "Simulated attack pattern for testing",
            "remediation_actions": [
                "Monitoring patterns updated",
                "Alert thresholds tuned",
            ],
            "false_positive": True,  # This was a test
            "lessons_learned": "System correctly identified anomalous patterns",
        }

        # In a real system, this would create a formal incident report
        # For testing, we'll verify the data was processed correctly
        assert len(incident_events) == 5
        assert sum(1 for event in incident_events if event["is_anomaly"]) >= 3

    def _get_algorithm_parameters(self, algorithm: str) -> dict:
        """Get default parameters for different algorithms."""
        parameters = {
            "isolation_forest": {
                "contamination": 0.1,
                "random_state": 42,
                "n_estimators": 100,
            },
            "one_class_svm": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"},
            "local_outlier_factor": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "algorithm": "auto",
            },
            "elliptic_envelope": {
                "contamination": 0.1,
                "random_state": 42,
                "support_fraction": None,
            },
            "time_series_anomaly": {
                "contamination": 0.05,
                "seasonal_period": 24,
                "trend_detection": True,
            },
        }

        return parameters.get(algorithm, {"contamination": 0.1, "random_state": 42})
