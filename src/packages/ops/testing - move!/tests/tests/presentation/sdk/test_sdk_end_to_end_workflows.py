"""
SDK End-to-End Workflows Testing
================================

This module provides comprehensive end-to-end testing for SDK workflows,
covering complete data pipelines from dataset upload to anomaly detection.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest


class TestCompleteDetectionWorkflow:
    """Test suite for complete anomaly detection workflows."""

    @pytest.fixture
    def workflow_client(self):
        """Create mock client for workflow testing."""
        client = Mock()
        client.base_url = "https://api.monorepo.com"
        client.headers = {"X-API-Key": "test-key"}
        return client

    @pytest.fixture
    def sample_workflow_data(self):
        """Create sample data for workflow testing."""
        # Generate synthetic dataset
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (900, 5))
        anomaly_data = np.random.normal(5, 2, (100, 5))  # Shifted anomalies

        data = np.vstack([normal_data, anomaly_data])
        labels = np.array([0] * 900 + [1] * 100)  # 0 = normal, 1 = anomaly

        return {
            "training_data": data[:800],
            "test_data": data[800:],
            "training_labels": labels[:800],
            "test_labels": labels[800:],
            "features": [f"feature_{i}" for i in range(5)],
        }

    def test_complete_isolation_forest_workflow(
        self, workflow_client, sample_workflow_data
    ):
        """Test complete workflow using Isolation Forest algorithm."""
        # Step 1: Upload training dataset
        training_dataset_response = {
            "id": "training-dataset-123",
            "name": "training_data",
            "status": "ready",
            "shape": [800, 5],
            "features": sample_workflow_data["features"],
        }
        workflow_client.create_dataset = Mock(return_value=training_dataset_response)

        dataset_result = workflow_client.create_dataset(
            name="training_data",
            data=sample_workflow_data["training_data"].tolist(),
            features=sample_workflow_data["features"],
        )

        assert dataset_result["id"] == "training-dataset-123"
        assert dataset_result["status"] == "ready"

        # Step 2: Create and train detector
        detector_response = {
            "id": "detector-456",
            "name": "isolation_forest_detector",
            "algorithm": "isolation_forest",
            "status": "training",
            "parameters": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
        }
        workflow_client.create_detector = Mock(return_value=detector_response)

        detector_result = workflow_client.create_detector(
            name="isolation_forest_detector",
            algorithm="isolation_forest",
            parameters={"n_estimators": 100, "contamination": 0.1, "random_state": 42},
            training_data="training-dataset-123",
        )

        assert detector_result["id"] == "detector-456"
        assert detector_result["algorithm"] == "isolation_forest"

        # Step 3: Monitor training progress
        training_status_responses = [
            {"status": "training", "progress": 25},
            {"status": "training", "progress": 50},
            {"status": "training", "progress": 75},
            {
                "status": "trained",
                "progress": 100,
                "performance_metrics": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.94,
                    "f1_score": 0.91,
                },
            },
        ]
        workflow_client.get_detector = Mock(side_effect=training_status_responses)

        # Poll training status
        for expected_response in training_status_responses:
            status = workflow_client.get_detector("detector-456")
            assert status["status"] == expected_response["status"]
            assert status["progress"] == expected_response["progress"]

        # Step 4: Upload test dataset
        test_dataset_response = {
            "id": "test-dataset-789",
            "name": "test_data",
            "status": "ready",
            "shape": [200, 5],
        }
        workflow_client.create_dataset = Mock(return_value=test_dataset_response)

        test_dataset_result = workflow_client.create_dataset(
            name="test_data",
            data=sample_workflow_data["test_data"].tolist(),
            features=sample_workflow_data["features"],
        )

        assert test_dataset_result["id"] == "test-dataset-789"

        # Step 5: Perform anomaly detection
        detection_response = {
            "detection_id": "detection-999",
            "detector_id": "detector-456",
            "dataset_id": "test-dataset-789",
            "predictions": [0] * 180 + [1] * 20,  # 20 anomalies detected
            "scores": np.random.random(200).tolist(),
            "execution_time": 0.152,
            "statistics": {
                "total_samples": 200,
                "anomalies_detected": 20,
                "anomaly_rate": 0.10,
            },
        }
        workflow_client.detect_anomalies = Mock(return_value=detection_response)

        detection_result = workflow_client.detect_anomalies(
            detector_id="detector-456", dataset_id="test-dataset-789"
        )

        assert detection_result["detection_id"] == "detection-999"
        assert detection_result["statistics"]["anomalies_detected"] == 20
        assert detection_result["statistics"]["anomaly_rate"] == 0.10

        # Verify all method calls
        assert workflow_client.create_dataset.call_count == 2
        assert workflow_client.create_detector.call_count == 1
        assert workflow_client.get_detector.call_count == 4
        assert workflow_client.detect_anomalies.call_count == 1

    def test_multi_algorithm_comparison_workflow(
        self, workflow_client, sample_workflow_data
    ):
        """Test workflow comparing multiple algorithms."""
        # Step 1: Create shared training dataset
        dataset_response = {
            "id": "shared-dataset-123",
            "name": "comparison_data",
            "status": "ready",
            "shape": [800, 5],
        }
        workflow_client.create_dataset = Mock(return_value=dataset_response)

        dataset_result = workflow_client.create_dataset(
            name="comparison_data",
            data=sample_workflow_data["training_data"].tolist(),
            features=sample_workflow_data["features"],
        )

        # Step 2: Create multiple detectors with different algorithms
        algorithms_config = [
            {
                "name": "isolation_forest",
                "params": {"n_estimators": 100, "contamination": 0.1},
            },
            {
                "name": "local_outlier_factor",
                "params": {"n_neighbors": 20, "contamination": 0.1},
            },
            {"name": "one_class_svm", "params": {"gamma": "scale", "nu": 0.1}},
            {"name": "elliptic_envelope", "params": {"contamination": 0.1}},
        ]

        detector_responses = []
        for i, config in enumerate(algorithms_config):
            response = {
                "id": f"detector-{i+1}",
                "name": f"{config['name']}_detector",
                "algorithm": config["name"],
                "status": "trained",
                "parameters": config["params"],
                "performance_metrics": {
                    "accuracy": np.random.uniform(0.85, 0.95),
                    "precision": np.random.uniform(0.80, 0.90),
                    "recall": np.random.uniform(0.85, 0.95),
                    "f1_score": np.random.uniform(0.82, 0.92),
                },
            }
            detector_responses.append(response)

        workflow_client.create_detector = Mock(side_effect=detector_responses)

        # Create all detectors
        created_detectors = []
        for config in algorithms_config:
            detector = workflow_client.create_detector(
                name=f"{config['name']}_detector",
                algorithm=config["name"],
                parameters=config["params"],
                training_data="shared-dataset-123",
            )
            created_detectors.append(detector)

        assert len(created_detectors) == 4
        assert workflow_client.create_detector.call_count == 4

        # Step 3: Run comparison experiment
        experiment_response = {
            "experiment_id": "experiment-777",
            "name": "algorithm_comparison",
            "detectors": [d["id"] for d in created_detectors],
            "test_dataset": "test-dataset-789",
            "status": "completed",
            "results": [
                {
                    "detector_id": detector["id"],
                    "algorithm": detector["algorithm"],
                    "performance": detector["performance_metrics"],
                    "detection_time": np.random.uniform(0.1, 0.5),
                }
                for detector in created_detectors
            ],
            "ranking": [
                {"detector_id": "detector-1", "rank": 1, "score": 0.91},
                {"detector_id": "detector-3", "rank": 2, "score": 0.89},
                {"detector_id": "detector-2", "rank": 3, "score": 0.87},
                {"detector_id": "detector-4", "rank": 4, "score": 0.85},
            ],
        }
        workflow_client.create_experiment = Mock(return_value=experiment_response)

        experiment_result = workflow_client.create_experiment(
            name="algorithm_comparison",
            detector_ids=[d["id"] for d in created_detectors],
            test_dataset="test-dataset-789",
            metrics=["accuracy", "precision", "recall", "f1_score", "detection_time"],
        )

        assert experiment_result["experiment_id"] == "experiment-777"
        assert len(experiment_result["results"]) == 4
        assert len(experiment_result["ranking"]) == 4
        assert experiment_result["ranking"][0]["rank"] == 1

    def test_batch_processing_workflow(self, workflow_client, sample_workflow_data):
        """Test large-scale batch processing workflow."""
        # Step 1: Create detector
        detector_response = {
            "id": "batch-detector-123",
            "name": "batch_processor",
            "algorithm": "isolation_forest",
            "status": "trained",
        }
        workflow_client.create_detector = Mock(return_value=detector_response)

        detector = workflow_client.create_detector(
            name="batch_processor",
            algorithm="isolation_forest",
            parameters={"n_estimators": 100, "contamination": 0.05},
        )

        # Step 2: Upload large dataset in chunks
        large_dataset_size = 50000
        chunk_size = 10000
        chunk_responses = []

        for i in range(0, large_dataset_size, chunk_size):
            chunk_response = {
                "id": f"chunk-{i//chunk_size}",
                "name": f"data_chunk_{i//chunk_size}",
                "status": "ready",
                "shape": [min(chunk_size, large_dataset_size - i), 5],
            }
            chunk_responses.append(chunk_response)

        workflow_client.create_dataset = Mock(side_effect=chunk_responses)

        # Upload chunks
        uploaded_chunks = []
        for i in range(len(chunk_responses)):
            chunk_data = np.random.random((chunk_size, 5))
            chunk = workflow_client.create_dataset(
                name=f"data_chunk_{i}",
                data=chunk_data.tolist(),
                features=sample_workflow_data["features"],
            )
            uploaded_chunks.append(chunk)

        assert len(uploaded_chunks) == 5  # 50000 / 10000

        # Step 3: Process chunks in batches
        batch_job_response = {
            "job_id": "batch-job-456",
            "detector_id": "batch-detector-123",
            "chunk_ids": [chunk["id"] for chunk in uploaded_chunks],
            "status": "running",
            "progress": {
                "total_chunks": 5,
                "completed_chunks": 0,
                "estimated_time": 300,
            },
        }
        workflow_client.start_batch_job = Mock(return_value=batch_job_response)

        batch_job = workflow_client.start_batch_job(
            detector_id="batch-detector-123",
            chunk_ids=[chunk["id"] for chunk in uploaded_chunks],
            batch_size=2,  # Process 2 chunks at a time
        )

        assert batch_job["job_id"] == "batch-job-456"
        assert batch_job["progress"]["total_chunks"] == 5

        # Step 4: Monitor batch job progress
        progress_responses = [
            {
                "status": "running",
                "progress": {"completed_chunks": 1, "total_chunks": 5},
            },
            {
                "status": "running",
                "progress": {"completed_chunks": 3, "total_chunks": 5},
            },
            {
                "status": "completed",
                "progress": {"completed_chunks": 5, "total_chunks": 5},
                "results": {
                    "total_samples_processed": 50000,
                    "total_anomalies_detected": 2500,
                    "overall_anomaly_rate": 0.05,
                    "processing_time": 245.7,
                },
            },
        ]
        workflow_client.get_batch_job_status = Mock(side_effect=progress_responses)

        # Poll job status
        for expected_response in progress_responses:
            status = workflow_client.get_batch_job_status("batch-job-456")
            assert status["status"] == expected_response["status"]
            if "results" in expected_response:
                assert status["results"]["total_samples_processed"] == 50000
                assert status["results"]["total_anomalies_detected"] == 2500

    @pytest.mark.asyncio
    async def test_real_time_streaming_workflow(
        self, workflow_client, sample_workflow_data
    ):
        """Test real-time streaming detection workflow."""
        # Convert sync client to async for this test
        async_client = AsyncMock()
        async_client.base_url = workflow_client.base_url
        async_client.headers = workflow_client.headers

        # Step 1: Create streaming detector
        detector_response = {
            "id": "streaming-detector-789",
            "name": "real_time_detector",
            "algorithm": "isolation_forest",
            "status": "deployed",
            "streaming_endpoint": "/stream/detector-789",
        }
        async_client.create_detector = AsyncMock(return_value=detector_response)

        detector = await async_client.create_detector(
            name="real_time_detector",
            algorithm="isolation_forest",
            parameters={
                "n_estimators": 50,
                "contamination": 0.1,
            },  # Faster for streaming
            enable_streaming=True,
        )

        assert detector["id"] == "streaming-detector-789"
        assert detector["status"] == "deployed"

        # Step 2: Set up streaming data source
        async def data_generator():
            """Generate streaming data samples."""
            for i in range(100):
                # Mix normal and anomalous samples
                if i % 20 == 0:
                    # Anomalous sample
                    sample = {
                        "id": i,
                        "timestamp": datetime.now().isoformat(),
                        "features": np.random.normal(
                            5, 2, 5
                        ).tolist(),  # Shifted distribution
                        "metadata": {"source": "sensor_A", "batch": i // 10},
                    }
                else:
                    # Normal sample
                    sample = {
                        "id": i,
                        "timestamp": datetime.now().isoformat(),
                        "features": np.random.normal(
                            0, 1, 5
                        ).tolist(),  # Normal distribution
                        "metadata": {"source": "sensor_A", "batch": i // 10},
                    }

                yield sample
                await asyncio.sleep(0.01)  # Simulate real-time arrival

        # Step 3: Process streaming data
        async def mock_stream_detection(detector_id, data_stream):
            """Mock streaming detection."""
            async for sample in data_stream:
                # Simulate detection processing
                await asyncio.sleep(0.005)

                # Generate prediction based on features
                features = np.array(sample["features"])
                is_anomaly = np.mean(features) > 2.0  # Simple threshold for mock

                result = {
                    "sample_id": sample["id"],
                    "timestamp": sample["timestamp"],
                    "prediction": 1 if is_anomaly else 0,
                    "score": np.random.uniform(0.8, 1.0)
                    if is_anomaly
                    else np.random.uniform(0.0, 0.3),
                    "processing_time": 0.005,
                    "detector_id": detector_id,
                }
                yield result

        async_client.stream_detection = mock_stream_detection

        # Process streaming data
        results = []
        anomalies_detected = 0

        async for result in async_client.stream_detection(
            "streaming-detector-789", data_generator()
        ):
            results.append(result)
            if result["prediction"] == 1:
                anomalies_detected += 1

            # Stop after collecting enough samples
            if len(results) >= 50:
                break

        assert len(results) == 50
        assert anomalies_detected > 0  # Should detect some anomalies
        assert all("sample_id" in r for r in results)
        assert all("processing_time" in r for r in results)

        # Step 4: Generate streaming summary
        streaming_summary = {
            "session_id": "stream-session-123",
            "detector_id": "streaming-detector-789",
            "total_samples_processed": len(results),
            "anomalies_detected": anomalies_detected,
            "anomaly_rate": anomalies_detected / len(results),
            "average_processing_time": np.mean([r["processing_time"] for r in results]),
            "session_duration": len(results) * 0.01,  # Simulated time
        }

        assert streaming_summary["total_samples_processed"] == 50
        assert 0 <= streaming_summary["anomaly_rate"] <= 1
        assert streaming_summary["average_processing_time"] > 0


class TestMLOpsWorkflows:
    """Test suite for MLOps workflows including model lifecycle management."""

    @pytest.fixture
    def mlops_client(self):
        """Create mock client for MLOps testing."""
        client = Mock()
        client.base_url = "https://api.monorepo.com"
        client.headers = {"X-API-Key": "mlops-key"}
        return client

    def test_model_versioning_workflow(self, mlops_client):
        """Test model versioning and deployment workflow."""
        # Step 1: Create initial model version
        model_v1_response = {
            "id": "model-v1-123",
            "name": "anomaly_detector",
            "version": "1.0.0",
            "algorithm": "isolation_forest",
            "status": "trained",
            "performance": {"accuracy": 0.87, "f1_score": 0.85},
            "created_at": "2023-01-01T10:00:00Z",
        }
        mlops_client.create_model = Mock(return_value=model_v1_response)

        model_v1 = mlops_client.create_model(
            name="anomaly_detector",
            version="1.0.0",
            algorithm="isolation_forest",
            parameters={"n_estimators": 100, "contamination": 0.1},
        )

        assert model_v1["version"] == "1.0.0"
        assert model_v1["performance"]["accuracy"] == 0.87

        # Step 2: Deploy v1 to staging
        deployment_v1_response = {
            "deployment_id": "deploy-v1-456",
            "model_id": "model-v1-123",
            "environment": "staging",
            "status": "deployed",
            "endpoint_url": "https://api.monorepo.com/v1/models/deploy-v1-456",
        }
        mlops_client.deploy_model = Mock(return_value=deployment_v1_response)

        staging_deployment = mlops_client.deploy_model(
            model_id="model-v1-123", environment="staging"
        )

        assert staging_deployment["environment"] == "staging"
        assert staging_deployment["status"] == "deployed"

        # Step 3: Create improved model version
        model_v2_response = {
            "id": "model-v2-789",
            "name": "anomaly_detector",
            "version": "2.0.0",
            "algorithm": "ensemble",  # Improved algorithm
            "status": "trained",
            "performance": {"accuracy": 0.92, "f1_score": 0.90},  # Better performance
            "created_at": "2023-02-01T10:00:00Z",
        }
        mlops_client.create_model = Mock(return_value=model_v2_response)

        model_v2 = mlops_client.create_model(
            name="anomaly_detector",
            version="2.0.0",
            algorithm="ensemble",
            parameters={
                "algorithms": ["isolation_forest", "local_outlier_factor"],
                "voting": "soft",
                "weights": [0.6, 0.4],
            },
        )

        assert model_v2["version"] == "2.0.0"
        assert model_v2["performance"]["accuracy"] > model_v1["performance"]["accuracy"]

        # Step 4: A/B test between versions
        ab_test_response = {
            "test_id": "ab-test-999",
            "name": "v1_vs_v2_comparison",
            "model_a": "model-v1-123",
            "model_b": "model-v2-789",
            "traffic_split": {"model_a": 0.5, "model_b": 0.5},
            "status": "running",
            "metrics": {
                "requests_model_a": 500,
                "requests_model_b": 500,
                "accuracy_model_a": 0.87,
                "accuracy_model_b": 0.92,
                "latency_model_a": 0.15,
                "latency_model_b": 0.18,
            },
        }
        mlops_client.create_ab_test = Mock(return_value=ab_test_response)

        ab_test = mlops_client.create_ab_test(
            name="v1_vs_v2_comparison",
            model_a="model-v1-123",
            model_b="model-v2-789",
            traffic_split=0.5,
            duration_hours=24,
        )

        assert ab_test["test_id"] == "ab-test-999"
        assert (
            ab_test["metrics"]["accuracy_model_b"]
            > ab_test["metrics"]["accuracy_model_a"]
        )

        # Step 5: Promote winning model to production
        promotion_response = {
            "deployment_id": "deploy-v2-prod-111",
            "model_id": "model-v2-789",
            "environment": "production",
            "status": "deployed",
            "traffic_percentage": 100,
            "rollout_strategy": "blue_green",
        }
        mlops_client.promote_model = Mock(return_value=promotion_response)

        production_deployment = mlops_client.promote_model(
            model_id="model-v2-789",
            environment="production",
            rollout_strategy="blue_green",
        )

        assert production_deployment["environment"] == "production"
        assert production_deployment["traffic_percentage"] == 100

    def test_model_monitoring_workflow(self, mlops_client):
        """Test model monitoring and drift detection workflow."""
        # Step 1: Deploy model with monitoring
        model_response = {
            "id": "monitored-model-456",
            "name": "production_detector",
            "version": "1.0.0",
            "status": "deployed",
            "monitoring": {
                "enabled": True,
                "drift_detection": True,
                "performance_tracking": True,
                "alert_thresholds": {
                    "accuracy_drop": 0.05,
                    "drift_score": 0.3,
                    "latency_increase": 2.0,
                },
            },
        }
        mlops_client.deploy_model_with_monitoring = Mock(return_value=model_response)

        deployed_model = mlops_client.deploy_model_with_monitoring(
            model_id="model-123",
            environment="production",
            monitoring_config={
                "drift_detection": True,
                "performance_tracking": True,
                "alert_frequency": "hourly",
            },
        )

        assert deployed_model["monitoring"]["enabled"] is True
        assert deployed_model["monitoring"]["drift_detection"] is True

        # Step 2: Simulate monitoring data collection
        monitoring_data = [
            {
                "timestamp": "2023-01-01T10:00:00Z",
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "drift_score": 0.12,
                "latency_p95": 0.15,
                "request_count": 1000,
            },
            {
                "timestamp": "2023-01-01T11:00:00Z",
                "accuracy": 0.90,  # Slight drop
                "precision": 0.87,
                "recall": 0.92,
                "drift_score": 0.18,
                "latency_p95": 0.16,
                "request_count": 1050,
            },
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "accuracy": 0.85,  # Significant drop - should trigger alert
                "precision": 0.82,
                "recall": 0.88,
                "drift_score": 0.35,  # High drift - should trigger alert
                "latency_p95": 0.45,  # High latency - should trigger alert
                "request_count": 980,
            },
        ]

        # Step 3: Process monitoring alerts
        alert_responses = []
        for data in monitoring_data:
            alerts = []

            # Check accuracy drop
            if data["accuracy"] < 0.92 - 0.05:  # Below threshold
                alerts.append(
                    {
                        "type": "accuracy_drop",
                        "severity": "high",
                        "message": f"Accuracy dropped to {data['accuracy']:.2f}",
                        "threshold": 0.87,
                        "current_value": data["accuracy"],
                    }
                )

            # Check drift score
            if data["drift_score"] > 0.3:
                alerts.append(
                    {
                        "type": "data_drift",
                        "severity": "medium",
                        "message": f"Data drift detected: {data['drift_score']:.2f}",
                        "threshold": 0.3,
                        "current_value": data["drift_score"],
                    }
                )

            # Check latency
            if data["latency_p95"] > 0.15 * 2.0:  # 2x increase
                alerts.append(
                    {
                        "type": "latency_increase",
                        "severity": "medium",
                        "message": f"Latency increased to {data['latency_p95']:.2f}s",
                        "threshold": 0.30,
                        "current_value": data["latency_p95"],
                    }
                )

            alert_response = {
                "timestamp": data["timestamp"],
                "model_id": "monitored-model-456",
                "alerts": alerts,
                "monitoring_data": data,
            }
            alert_responses.append(alert_response)

        # Verify alerts are generated correctly
        assert len(alert_responses) == 3
        assert len(alert_responses[0]["alerts"]) == 0  # No alerts for first hour
        assert len(alert_responses[1]["alerts"]) == 0  # Minor degradation, no alerts
        assert len(alert_responses[2]["alerts"]) == 3  # All thresholds exceeded

        # Step 4: Trigger model retraining
        critical_alerts = [
            alert
            for alert in alert_responses[2]["alerts"]
            if alert["severity"] == "high"
        ]
        if critical_alerts:
            retraining_response = {
                "retraining_job_id": "retrain-job-789",
                "model_id": "monitored-model-456",
                "trigger_reason": "accuracy_degradation",
                "status": "queued",
                "estimated_duration": 1800,  # 30 minutes
            }
            mlops_client.trigger_retraining = Mock(return_value=retraining_response)

            retraining_job = mlops_client.trigger_retraining(
                model_id="monitored-model-456",
                trigger_reason="accuracy_degradation",
                use_recent_data=True,
                retrain_percentage=0.2,  # Retrain with 20% new data
            )

            assert retraining_job["retraining_job_id"] == "retrain-job-789"
            assert retraining_job["trigger_reason"] == "accuracy_degradation"

    def test_automated_pipeline_workflow(self, mlops_client):
        """Test automated ML pipeline workflow."""
        # Step 1: Configure automated pipeline
        pipeline_config = {
            "name": "automated_anomaly_detection",
            "trigger": {
                "type": "schedule",
                "cron": "0 2 * * *",  # Daily at 2 AM
                "timezone": "UTC",
            },
            "data_source": {
                "type": "database",
                "connection": "prod_db",
                "query": "SELECT * FROM sensor_data WHERE created_at >= NOW() - INTERVAL 1 DAY",
            },
            "preprocessing": {
                "steps": ["remove_nulls", "normalize", "feature_selection"],
                "feature_selection_method": "variance_threshold",
                "normalization_method": "standard_scaler",
            },
            "training": {
                "algorithms": ["isolation_forest", "local_outlier_factor", "ensemble"],
                "hyperparameter_tuning": True,
                "cross_validation_folds": 5,
                "model_selection_metric": "f1_score",
            },
            "deployment": {
                "auto_deploy": True,
                "deployment_strategy": "canary",
                "canary_percentage": 10,
                "success_criteria": {
                    "accuracy_threshold": 0.90,
                    "latency_threshold": 0.20,
                },
            },
        }

        pipeline_response = {
            "pipeline_id": "pipeline-auto-123",
            "name": "automated_anomaly_detection",
            "status": "active",
            "next_run": "2023-01-02T02:00:00Z",
            "config": pipeline_config,
        }
        mlops_client.create_pipeline = Mock(return_value=pipeline_response)

        pipeline = mlops_client.create_pipeline(config=pipeline_config)

        assert pipeline["pipeline_id"] == "pipeline-auto-123"
        assert pipeline["status"] == "active"

        # Step 2: Simulate pipeline execution
        execution_steps = [
            {
                "step": "data_extraction",
                "status": "completed",
                "duration": 45,
                "output": {
                    "rows_extracted": 50000,
                    "features": 12,
                    "data_quality_score": 0.95,
                },
            },
            {
                "step": "preprocessing",
                "status": "completed",
                "duration": 120,
                "output": {
                    "rows_after_cleaning": 48500,
                    "features_selected": 8,
                    "normalization_applied": True,
                },
            },
            {
                "step": "model_training",
                "status": "completed",
                "duration": 900,
                "output": {
                    "models_trained": 3,
                    "best_model": "isolation_forest",
                    "best_score": 0.932,
                    "hyperparameters": {"n_estimators": 150, "contamination": 0.08},
                },
            },
            {
                "step": "model_validation",
                "status": "completed",
                "duration": 180,
                "output": {
                    "validation_accuracy": 0.928,
                    "validation_precision": 0.91,
                    "validation_recall": 0.94,
                    "passes_criteria": True,
                },
            },
            {
                "step": "deployment",
                "status": "completed",
                "duration": 300,
                "output": {
                    "deployment_id": "auto-deploy-456",
                    "canary_percentage": 10,
                    "canary_status": "healthy",
                    "full_rollout_scheduled": "2023-01-02T06:00:00Z",
                },
            },
        ]

        pipeline_execution_response = {
            "execution_id": "exec-789",
            "pipeline_id": "pipeline-auto-123",
            "started_at": "2023-01-02T02:00:00Z",
            "status": "completed",
            "total_duration": 1545,  # Sum of all step durations
            "steps": execution_steps,
        }
        mlops_client.get_pipeline_execution = Mock(
            return_value=pipeline_execution_response
        )

        execution_result = mlops_client.get_pipeline_execution("exec-789")

        assert execution_result["status"] == "completed"
        assert len(execution_result["steps"]) == 5
        assert all(step["status"] == "completed" for step in execution_result["steps"])

        # Verify best model performance meets criteria
        training_output = execution_result["steps"][2]["output"]
        validation_output = execution_result["steps"][3]["output"]

        assert training_output["best_score"] > 0.90
        assert validation_output["validation_accuracy"] > 0.90
        assert validation_output["passes_criteria"] is True

        # Verify deployment was successful
        deployment_output = execution_result["steps"][4]["output"]
        assert deployment_output["canary_status"] == "healthy"
        assert "full_rollout_scheduled" in deployment_output
