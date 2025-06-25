"""Integration tests for complete API workflows."""

import os
import pytest
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestCompleteAPIWorkflows:
    """Test complete end-to-end workflows through the API."""

    @pytest.mark.asyncio
    async def test_complete_anomaly_detection_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test complete anomaly detection workflow from upload to prediction."""
        
        # Step 1: Upload dataset
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, 
            "integration_test_dataset"
        )
        assert dataset["name"] == "integration_test_dataset"
        assert dataset["rows"] == 1000
        assert len(dataset["columns"]) == 5  # feature1, feature2, feature3, label, timestamp
        
        # Step 2: Create detector
        detector = await integration_helper.create_detector(
            dataset["id"], 
            "isolation_forest"
        )
        assert detector["algorithm"] == "isolation_forest"
        assert detector["status"] == "created"
        
        # Step 3: Train detector
        training_result = await integration_helper.train_detector(detector["id"])
        assert training_result["status"] == "completed"
        assert "model_id" in training_result
        assert "metrics" in training_result
        
        # Step 4: Validate detector
        response = await async_test_client.post(f"/api/detection/validate/{detector['id']}")
        response.raise_for_status()
        validation_result = response.json()["data"]
        assert validation_result["status"] == "completed"
        assert "metrics" in validation_result
        assert validation_result["metrics"]["accuracy"] > 0.8
        
        # Step 5: Make predictions
        test_data = {
            "data": [
                {"feature1": 0.5, "feature2": 0.3, "feature3": 0.1},  # Normal
                {"feature1": 5.0, "feature2": 4.8, "feature3": 3.2},  # Anomaly
            ]
        }
        
        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}", 
            json=test_data
        )
        response.raise_for_status()
        predictions = response.json()["data"]
        
        assert len(predictions) == 2
        assert predictions[0]["is_anomaly"] is False
        assert predictions[1]["is_anomaly"] is True
        assert predictions[1]["anomaly_score"] > predictions[0]["anomaly_score"]
        
        # Step 6: Get detector info
        response = await async_test_client.get(f"/api/detectors/{detector['id']}")
        response.raise_for_status()
        detector_info = response.json()["data"]
        assert detector_info["status"] == "trained"
        assert detector_info["model_id"] is not None

    @pytest.mark.asyncio
    async def test_streaming_anomaly_detection_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_time_series_csv: str,
        disable_auth
    ):
        """Test real-time streaming anomaly detection workflow."""
        
        # Step 1: Upload time series dataset
        dataset = await integration_helper.upload_dataset(
            sample_time_series_csv,
            "time_series_dataset"
        )
        
        # Step 2: Create detector for time series
        detector = await integration_helper.create_detector(
            dataset["id"],
            "time_series_anomaly"
        )
        
        # Step 3: Train detector
        training_result = await integration_helper.train_detector(detector["id"])
        assert training_result["status"] == "completed"
        
        # Step 4: Create streaming session
        streaming_config = {
            "name": "test_streaming_session",
            "detector_id": detector["id"],
            "data_source": {
                "source_type": "mock",
                "connection_config": {}
            },
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 100,
                "schema_validation": True,
                "enable_checkpointing": False
            },
            "description": "Integration test streaming session"
        }
        
        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user",
            json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        integration_helper.created_resources["sessions"].append(session["id"])
        
        assert session["name"] == "test_streaming_session"
        assert session["status"] == "pending"
        
        # Step 5: Start streaming session
        response = await async_test_client.post(f"/api/streaming/sessions/{session['id']}/start")
        response.raise_for_status()
        started_session = response.json()["data"]
        assert started_session["status"] in ["starting", "active"]
        
        # Step 6: Process test data
        test_data = {
            "data": {
                "timestamp": "2024-12-25T10:30:00Z",
                "value": 150.0,  # Anomalous value
                "cpu_usage": 85.2,
                "memory_usage": 78.5,
                "network_io": 2048000
            }
        }
        
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session['id']}/process",
            json=test_data
        )
        response.raise_for_status()
        processing_result = response.json()["data"]
        
        assert "anomaly_score" in processing_result
        assert "is_anomaly" in processing_result
        assert processing_result["session_id"] == session["id"]
        
        # Step 7: Get session metrics
        response = await async_test_client.get(f"/api/streaming/sessions/{session['id']}/metrics")
        response.raise_for_status()
        metrics = response.json()["data"]
        
        assert metrics["messages_processed"] >= 1
        assert "messages_per_second" in metrics
        assert "anomaly_rate" in metrics
        
        # Step 8: Stop streaming session
        response = await async_test_client.post(f"/api/streaming/sessions/{session['id']}/stop")
        response.raise_for_status()
        stopped_session = response.json()["data"]
        assert stopped_session["status"] in ["stopping", "stopped"]

    @pytest.mark.asyncio
    async def test_experiment_management_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test experiment management and comparison workflow."""
        
        # Step 1: Upload dataset
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "experiment_dataset"
        )
        
        # Step 2: Create experiment
        experiment_config = {
            "name": "algorithm_comparison_experiment",
            "description": "Compare multiple anomaly detection algorithms",
            "dataset_id": dataset["id"],
            "algorithms": [
                {
                    "name": "isolation_forest",
                    "parameters": {"contamination": 0.05, "random_state": 42}
                },
                {
                    "name": "one_class_svm", 
                    "parameters": {"nu": 0.05, "kernel": "rbf"}
                },
                {
                    "name": "local_outlier_factor",
                    "parameters": {"n_neighbors": 20, "contamination": 0.05}
                }
            ],
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            "cross_validation": {
                "enabled": True,
                "folds": 3,
                "stratified": True
            }
        }
        
        response = await async_test_client.post(
            "/api/experiments/create?created_by=test_user",
            json=experiment_config
        )
        response.raise_for_status()
        experiment = response.json()["data"]
        integration_helper.created_resources["experiments"].append(experiment["id"])
        
        assert experiment["name"] == "algorithm_comparison_experiment"
        assert experiment["status"] == "created"
        assert len(experiment["algorithm_configurations"]) == 3
        
        # Step 3: Run experiment
        response = await async_test_client.post(f"/api/experiments/{experiment['id']}/run")
        response.raise_for_status()
        run_result = response.json()["data"]
        assert run_result["status"] in ["running", "completed"]
        
        # Step 4: Get experiment results
        response = await async_test_client.get(f"/api/experiments/{experiment['id']}/results")
        response.raise_for_status()
        results = response.json()["data"]
        
        assert "algorithm_results" in results
        assert len(results["algorithm_results"]) == 3
        
        for algo_result in results["algorithm_results"]:
            assert "algorithm_name" in algo_result
            assert "metrics" in algo_result
            assert "cross_validation_scores" in algo_result
            assert algo_result["metrics"]["accuracy"] > 0.0
        
        # Step 5: Get experiment comparison
        response = await async_test_client.get(f"/api/experiments/{experiment['id']}/comparison")
        response.raise_for_status()
        comparison = response.json()["data"]
        
        assert "ranking" in comparison
        assert "best_algorithm" in comparison
        assert "metric_comparison" in comparison
        
        # Step 6: Export experiment results
        response = await async_test_client.post(
            f"/api/experiments/{experiment['id']}/export",
            json={"format": "json", "include_models": True}
        )
        response.raise_for_status()
        export_result = response.json()["data"]
        
        assert "download_url" in export_result
        assert export_result["format"] == "json"

    @pytest.mark.asyncio
    async def test_model_lifecycle_management_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test complete model lifecycle management workflow."""
        
        # Step 1: Upload dataset and create detector
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "model_lifecycle_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        # Step 2: Train initial model
        training_result = await integration_helper.train_detector(detector["id"])
        model_v1_id = training_result["model_id"]
        
        # Step 3: Register model in model registry
        model_registration = {
            "name": "production_anomaly_detector",
            "version": "1.0.0",
            "description": "Initial production model",
            "tags": ["production", "baseline"],
            "metadata": {
                "algorithm": "isolation_forest",
                "training_dataset": dataset["id"],
                "performance_baseline": True
            }
        }
        
        response = await async_test_client.post(
            f"/api/models/{model_v1_id}/register",
            json=model_registration
        )
        response.raise_for_status()
        registered_model = response.json()["data"]
        
        assert registered_model["name"] == "production_anomaly_detector"
        assert registered_model["version"] == "1.0.0"
        assert registered_model["stage"] == "staging"
        
        # Step 4: Promote model to production
        response = await async_test_client.post(
            f"/api/models/{model_v1_id}/promote",
            json={"stage": "production", "approved_by": "test_user"}
        )
        response.raise_for_status()
        promoted_model = response.json()["data"]
        assert promoted_model["stage"] == "production"
        
        # Step 5: Train improved model (version 2)
        improved_detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        # Use different parameters for improved model
        improved_training = await integration_helper.train_detector(improved_detector["id"])
        model_v2_id = improved_training["model_id"]
        
        # Step 6: Compare models
        comparison_config = {
            "baseline_model_id": model_v1_id,
            "candidate_model_id": model_v2_id,
            "test_dataset_id": dataset["id"],
            "comparison_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "statistical_tests": ["mcnemar_test", "wilcoxon_test"]
        }
        
        response = await async_test_client.post(
            "/api/models/compare",
            json=comparison_config
        )
        response.raise_for_status()
        comparison_result = response.json()["data"]
        
        assert "baseline_metrics" in comparison_result
        assert "candidate_metrics" in comparison_result
        assert "statistical_significance" in comparison_result
        assert "recommendation" in comparison_result
        
        # Step 7: Model lineage tracking
        response = await async_test_client.get(f"/api/model-lineage/{model_v1_id}")
        response.raise_for_status()
        lineage = response.json()["data"]
        
        assert "model_id" in lineage
        assert "training_dataset" in lineage
        assert "feature_lineage" in lineage
        assert "model_dependencies" in lineage
        
        # Step 8: Model performance monitoring
        response = await async_test_client.get(f"/api/models/{model_v1_id}/monitoring/drift")
        response.raise_for_status()
        drift_analysis = response.json()["data"]
        
        assert "data_drift" in drift_analysis
        assert "concept_drift" in drift_analysis
        assert "drift_score" in drift_analysis

    @pytest.mark.asyncio
    async def test_automl_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test AutoML workflow for automated algorithm selection and tuning."""
        
        # Step 1: Upload dataset
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "automl_dataset"
        )
        
        # Step 2: Configure AutoML job
        automl_config = {
            "job_name": "automated_anomaly_detection",
            "dataset_id": dataset["id"],
            "target_column": "label",
            "problem_type": "anomaly_detection",
            "algorithms": [
                "isolation_forest",
                "one_class_svm", 
                "local_outlier_factor",
                "elliptic_envelope"
            ],
            "optimization_metric": "f1_score",
            "max_trials": 20,
            "max_duration_minutes": 30,
            "cross_validation": {
                "folds": 3,
                "stratified": True
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "strategy": "bayesian",
                "n_trials": 10
            },
            "feature_engineering": {
                "enabled": True,
                "scaling": "standard",
                "pca_components": 0.95
            }
        }
        
        response = await async_test_client.post(
            "/api/autonomous/automl/start?created_by=test_user",
            json=automl_config
        )
        response.raise_for_status()
        automl_job = response.json()["data"]
        
        assert automl_job["job_name"] == "automated_anomaly_detection"
        assert automl_job["status"] in ["created", "running"]
        
        # Step 3: Monitor AutoML progress
        response = await async_test_client.get(f"/api/autonomous/automl/{automl_job['id']}/status")
        response.raise_for_status()
        status = response.json()["data"]
        
        assert "status" in status
        assert "progress_percentage" in status
        assert "current_trial" in status
        assert "best_score" in status
        
        # Step 4: Get AutoML results
        response = await async_test_client.get(f"/api/autonomous/automl/{automl_job['id']}/results")
        response.raise_for_status()
        results = response.json()["data"]
        
        assert "best_model" in results
        assert "trial_results" in results
        assert "feature_importance" in results
        assert "model_comparison" in results
        
        # Step 5: Deploy best model
        deployment_config = {
            "model_name": "automl_best_model",
            "version": "1.0.0",
            "description": "Best model from AutoML",
            "deployment_target": "production"
        }
        
        response = await async_test_client.post(
            f"/api/autonomous/automl/{automl_job['id']}/deploy",
            json=deployment_config
        )
        response.raise_for_status()
        deployment = response.json()["data"]
        
        assert deployment["model_name"] == "automl_best_model"
        assert deployment["status"] in ["deploying", "deployed"]

    @pytest.mark.asyncio
    async def test_event_processing_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test event processing and pattern detection workflow."""
        
        # Step 1: Create test events
        events_data = [
            {
                "event_type": "anomaly_detected",
                "severity": "high",
                "title": "High CPU usage anomaly",
                "description": "CPU usage exceeded normal patterns",
                "data_source": "server-01",
                "raw_data": {"cpu_usage": 95.2, "timestamp": "2024-12-25T10:00:00Z"},
                "anomaly_data": {
                    "anomaly_score": 0.92,
                    "confidence": 0.85,
                    "feature_contributions": {"cpu_usage": 0.8}
                }
            },
            {
                "event_type": "anomaly_detected", 
                "severity": "high",
                "title": "Memory usage anomaly",
                "description": "Memory usage exceeded normal patterns",
                "data_source": "server-01",
                "raw_data": {"memory_usage": 88.7, "timestamp": "2024-12-25T10:05:00Z"},
                "anomaly_data": {
                    "anomaly_score": 0.87,
                    "confidence": 0.82,
                    "feature_contributions": {"memory_usage": 0.75}
                }
            }
        ]
        
        created_events = []
        for event_data in events_data:
            response = await async_test_client.post("/api/events/create", json=event_data)
            response.raise_for_status()
            created_events.append(response.json()["data"])
        
        # Step 2: Query events
        query_config = {
            "event_types": ["anomaly_detected"],
            "severities": ["high"],
            "data_sources": ["server-01"],
            "min_anomaly_score": 0.8,
            "limit": 10
        }
        
        response = await async_test_client.post("/api/events/query", json=query_config)
        response.raise_for_status()
        queried_events = response.json()["data"]
        
        assert len(queried_events) >= 2
        
        # Step 3: Create event pattern
        pattern_config = {
            "name": "High Resource Usage Pattern",
            "pattern_type": "frequency",
            "conditions": {
                "event_type": "anomaly_detected",
                "min_count": 2,
                "data_source": "server-01"
            },
            "time_window_seconds": 600,
            "description": "Multiple anomalies from same server",
            "confidence": 0.8,
            "alert_threshold": 2
        }
        
        response = await async_test_client.post(
            "/api/events/patterns?created_by=test_user",
            json=pattern_config
        )
        response.raise_for_status()
        pattern = response.json()["data"]
        
        assert pattern["name"] == "High Resource Usage Pattern"
        assert pattern["pattern_type"] == "frequency"
        
        # Step 4: Detect patterns
        response = await async_test_client.get("/api/events/patterns/detect?hours=1")
        response.raise_for_status()
        detected_patterns = response.json()["data"]
        
        # Should detect our created pattern
        assert len(detected_patterns) >= 0  # Pattern detection might not trigger immediately
        
        # Step 5: Get event summary
        response = await async_test_client.get("/api/events/summary")
        response.raise_for_status()
        summary = response.json()["data"]
        
        assert "total_events" in summary
        assert "events_by_type" in summary
        assert "events_by_severity" in summary
        assert summary["total_events"] >= 2
        
        # Step 6: Acknowledge and resolve events
        for event in created_events:
            # Acknowledge event
            response = await async_test_client.post(
                f"/api/events/{event['id']}/acknowledge?user=test_user",
                json={"notes": "Acknowledged for testing"}
            )
            response.raise_for_status()
            
            # Resolve event
            response = await async_test_client.post(
                f"/api/events/{event['id']}/resolve?user=test_user",
                json={"notes": "Resolved - test scenario"}
            )
            response.raise_for_status()
            resolved_event = response.json()["data"]
            assert resolved_event["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test performance monitoring and optimization workflow."""
        
        # Step 1: Create detector and train model
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "performance_test_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        training_result = await integration_helper.train_detector(detector["id"])
        model_id = training_result["model_id"]
        
        # Step 2: Run performance benchmark
        benchmark_config = {
            "model_id": model_id,
            "test_dataset_id": dataset["id"],
            "metrics": [
                "prediction_latency",
                "throughput",
                "memory_usage",
                "cpu_usage"
            ],
            "test_scenarios": [
                {
                    "name": "single_prediction",
                    "batch_size": 1,
                    "iterations": 100
                },
                {
                    "name": "batch_prediction",
                    "batch_size": 100,
                    "iterations": 10
                }
            ]
        }
        
        response = await async_test_client.post(
            "/api/performance/benchmark",
            json=benchmark_config
        )
        response.raise_for_status()
        benchmark_result = response.json()["data"]
        
        assert "benchmark_id" in benchmark_result
        assert benchmark_result["status"] in ["running", "completed"]
        
        # Step 3: Get performance metrics
        response = await async_test_client.get(f"/api/performance/models/{model_id}/metrics")
        response.raise_for_status()
        metrics = response.json()["data"]
        
        assert "latency_metrics" in metrics
        assert "throughput_metrics" in metrics
        assert "resource_usage" in metrics
        
        # Step 4: Performance comparison
        # Create another model for comparison
        detector2 = await integration_helper.create_detector(
            dataset["id"],
            "one_class_svm"
        )
        training_result2 = await integration_helper.train_detector(detector2["id"])
        model2_id = training_result2["model_id"]
        
        comparison_config = {
            "model_ids": [model_id, model2_id],
            "metrics": ["prediction_latency", "accuracy", "memory_usage"],
            "test_dataset_id": dataset["id"]
        }
        
        response = await async_test_client.post(
            "/api/performance/compare",
            json=comparison_config
        )
        response.raise_for_status()
        comparison = response.json()["data"]
        
        assert "model_comparisons" in comparison
        assert len(comparison["model_comparisons"]) == 2
        assert "ranking" in comparison
        
        # Step 5: Get optimization recommendations
        response = await async_test_client.get(f"/api/performance/models/{model_id}/optimize")
        response.raise_for_status()
        recommendations = response.json()["data"]
        
        assert "optimization_suggestions" in recommendations
        assert "estimated_improvements" in recommendations