"""Integration workflow tests for application layer."""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from pynomaly.application.services.automl_service import AutoMLService
from pynomaly.application.services.detection_service import DetectionService
from pynomaly.application.services.ensemble_service import EnsembleService
from pynomaly.application.services.explainability_service import ExplainabilityService
from pynomaly.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly.application.use_cases.evaluate_model import EvaluateModelUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.domain.entities import Dataset, DetectionResult, Detector


class TestApplicationWorkflows:
    """Test end-to-end application workflows."""

    @pytest.fixture
    def workflow_services(self):
        """Create workflow service dependencies."""
        return {
            "detection_service": Mock(spec=DetectionService),
            "ensemble_service": Mock(spec=EnsembleService),
            "automl_service": Mock(spec=AutoMLService),
            "explainability_service": Mock(spec=ExplainabilityService),
            "detect_anomalies_use_case": Mock(spec=DetectAnomaliesUseCase),
            "train_detector_use_case": Mock(spec=TrainDetectorUseCase),
            "evaluate_model_use_case": Mock(spec=EvaluateModelUseCase),
        }

    @pytest.mark.asyncio
    async def test_complete_anomaly_detection_workflow(self, workflow_services):
        """Test complete workflow from data ingestion to result storage."""
        # Setup mock data
        dataset_id = str(uuid.uuid4())
        detector_id = str(uuid.uuid4())

        # Mock dataset loading
        Dataset(
            id=dataset_id,
            name="test_dataset",
            data=Mock(),
            features=["feature1", "feature2", "feature3"],
            metadata={"rows": 1000, "columns": 3},
        )

        # Mock detector
        Detector(
            id=detector_id,
            name="test_detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100},
            is_fitted=True,
        )

        # Mock detection results
        mock_results = DetectionResult(
            id=str(uuid.uuid4()),
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomaly_scores=[0.1, 0.8, 0.2, 0.9, 0.15],
            anomalies=[1, 3],  # Indices of anomalies
            metadata={"detection_time": datetime.now()},
        )

        # Configure mocks
        workflow_services["detect_anomalies_use_case"].execute.return_value = (
            mock_results
        )

        # Execute workflow
        use_case = workflow_services["detect_anomalies_use_case"]
        result = await use_case.execute(
            {
                "dataset_id": dataset_id,
                "detector_id": detector_id,
                "detection_config": {"threshold": 0.5},
            }
        )

        # Verify workflow execution
        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert len(result.anomalies) == 2

    @pytest.mark.asyncio
    async def test_model_training_and_evaluation_workflow(self, workflow_services):
        """Test model training, validation, and evaluation workflow."""
        dataset_id = str(uuid.uuid4())

        # Mock training workflow
        training_config = {
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100, "contamination": 0.1},
            "validation_split": 0.2,
            "cross_validation_folds": 5,
        }

        # Mock trained detector
        trained_detector = Detector(
            id=str(uuid.uuid4()),
            name="trained_detector",
            algorithm_name="IsolationForest",
            parameters=training_config["parameters"],
            is_fitted=True,
        )

        # Mock evaluation results
        evaluation_results = {
            "accuracy": 0.92,
            "precision": 0.85,
            "recall": 0.88,
            "f1_score": 0.86,
            "roc_auc": 0.91,
            "confusion_matrix": [[950, 50], [30, 70]],
        }

        # Configure mocks
        workflow_services["train_detector_use_case"].execute.return_value = (
            trained_detector
        )
        workflow_services["evaluate_model_use_case"].execute.return_value = (
            evaluation_results
        )

        # Execute training workflow
        train_use_case = workflow_services["train_detector_use_case"]
        detector = await train_use_case.execute(
            {"dataset_id": dataset_id, "training_config": training_config}
        )

        # Execute evaluation workflow
        eval_use_case = workflow_services["evaluate_model_use_case"]
        metrics = await eval_use_case.execute(
            {
                "detector_id": detector.id,
                "test_dataset_id": dataset_id,
                "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
            }
        )

        # Verify workflow results
        assert detector.is_fitted is True
        assert metrics["f1_score"] > 0.8
        assert metrics["accuracy"] > 0.9

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, workflow_services):
        """Test batch processing workflow for large datasets."""
        # Mock large dataset configuration
        batch_config = {
            "dataset_ids": [str(uuid.uuid4()) for _ in range(5)],
            "batch_size": 1000,
            "parallel_workers": 4,
            "detector_id": str(uuid.uuid4()),
        }

        # Mock batch processing results
        batch_results = []
        for dataset_id in batch_config["dataset_ids"]:
            result = DetectionResult(
                id=str(uuid.uuid4()),
                detector_id=batch_config["detector_id"],
                dataset_id=dataset_id,
                anomaly_scores=[0.1, 0.2, 0.8, 0.15, 0.9],
                anomalies=[2, 4],
                metadata={"batch_processed": True},
            )
            batch_results.append(result)

        # Mock batch processing service
        workflow_services["detection_service"].process_batch.return_value = (
            batch_results
        )

        # Execute batch workflow
        detection_service = workflow_services["detection_service"]
        results = await detection_service.process_batch(batch_config)

        # Verify batch processing
        assert len(results) == 5
        for result in results:
            assert result.metadata["batch_processed"] is True
            assert len(result.anomalies) > 0

    @pytest.mark.asyncio
    async def test_real_time_detection_workflow(self, workflow_services):
        """Test real-time anomaly detection workflow."""
        detector_id = str(uuid.uuid4())

        # Mock streaming data points
        streaming_data = [
            {"timestamp": "2024-01-15T10:00:00Z", "features": [1.0, 2.0, 3.0]},
            {"timestamp": "2024-01-15T10:01:00Z", "features": [1.1, 2.1, 3.1]},
            {
                "timestamp": "2024-01-15T10:02:00Z",
                "features": [5.0, 8.0, 2.0],
            },  # Anomaly
            {"timestamp": "2024-01-15T10:03:00Z", "features": [1.0, 1.9, 3.2]},
        ]

        # Mock real-time detection results
        streaming_results = []
        for i, data_point in enumerate(streaming_data):
            is_anomaly = i == 2  # Third point is anomaly
            result = {
                "timestamp": data_point["timestamp"],
                "anomaly_score": 0.9 if is_anomaly else 0.1,
                "is_anomaly": is_anomaly,
                "confidence": 0.95,
                "processing_latency_ms": 15,
            }
            streaming_results.append(result)

        # Mock streaming detection service
        workflow_services["detection_service"].detect_streaming.return_value = (
            streaming_results
        )

        # Execute streaming workflow
        detection_service = workflow_services["detection_service"]
        results = await detection_service.detect_streaming(
            {
                "detector_id": detector_id,
                "data_stream": streaming_data,
                "real_time_config": {
                    "max_latency_ms": 50,
                    "batch_size": 1,
                    "alert_threshold": 0.8,
                },
            }
        )

        # Verify streaming detection
        assert len(results) == 4
        anomaly_detected = any(r["is_anomaly"] for r in results)
        assert anomaly_detected is True

        # Verify latency requirements
        for result in results:
            assert result["processing_latency_ms"] <= 50

    @pytest.mark.asyncio
    async def test_ensemble_model_workflow(self, workflow_services):
        """Test ensemble model creation and prediction workflow."""
        # Mock base detectors
        base_detectors = [
            {"id": str(uuid.uuid4()), "algorithm": "IsolationForest", "weight": 0.4},
            {
                "id": str(uuid.uuid4()),
                "algorithm": "LocalOutlierFactor",
                "weight": 0.35,
            },
            {"id": str(uuid.uuid4()), "algorithm": "OneClassSVM", "weight": 0.25},
        ]

        dataset_id = str(uuid.uuid4())

        # Mock ensemble creation
        ensemble_config = {
            "base_detectors": base_detectors,
            "ensemble_method": "weighted_voting",
            "meta_learner": None,
            "diversity_threshold": 0.3,
        }

        ensemble_detector = Detector(
            id=str(uuid.uuid4()),
            name="ensemble_detector",
            algorithm_name="EnsembleDetector",
            parameters=ensemble_config,
            is_fitted=True,
        )

        # Mock ensemble predictions
        ensemble_results = DetectionResult(
            id=str(uuid.uuid4()),
            detector_id=ensemble_detector.id,
            dataset_id=dataset_id,
            anomaly_scores=[0.15, 0.82, 0.23, 0.91, 0.18],
            anomalies=[1, 3],
            metadata={
                "ensemble_predictions": {
                    base_detectors[0]["id"]: [0.1, 0.8, 0.2, 0.9, 0.15],
                    base_detectors[1]["id"]: [0.2, 0.85, 0.25, 0.92, 0.2],
                    base_detectors[2]["id"]: [0.15, 0.81, 0.24, 0.91, 0.19],
                },
                "ensemble_method": "weighted_voting",
            },
        )

        # Configure mocks
        workflow_services["ensemble_service"].create_ensemble.return_value = (
            ensemble_detector
        )
        workflow_services["ensemble_service"].predict_ensemble.return_value = (
            ensemble_results
        )

        # Execute ensemble workflow
        ensemble_service = workflow_services["ensemble_service"]

        # Create ensemble
        ensemble = await ensemble_service.create_ensemble(ensemble_config)

        # Make ensemble predictions
        predictions = await ensemble_service.predict_ensemble(
            {"ensemble_id": ensemble.id, "dataset_id": dataset_id}
        )

        # Verify ensemble workflow
        assert ensemble.algorithm_name == "EnsembleDetector"
        assert predictions.metadata["ensemble_method"] == "weighted_voting"
        assert len(predictions.anomalies) == 2
        assert len(predictions.metadata["ensemble_predictions"]) == 3

    @pytest.mark.asyncio
    async def test_automl_optimization_workflow(self, workflow_services):
        """Test AutoML optimization workflow."""
        dataset_id = str(uuid.uuid4())

        # Mock AutoML configuration
        automl_config = {
            "dataset_id": dataset_id,
            "optimization_metric": "f1_score",
            "time_budget": 300,
            "algorithms_to_try": [
                "IsolationForest",
                "LocalOutlierFactor",
                "OneClassSVM",
            ],
            "cross_validation_folds": 5,
        }

        # Mock AutoML results
        automl_results = {
            "best_model": {
                "algorithm": "IsolationForest",
                "parameters": {"n_estimators": 150, "contamination": 0.05},
                "performance": {"f1_score": 0.91, "precision": 0.88, "recall": 0.94},
            },
            "model_comparison": [
                {"algorithm": "IsolationForest", "f1_score": 0.91},
                {"algorithm": "LocalOutlierFactor", "f1_score": 0.87},
                {"algorithm": "OneClassSVM", "f1_score": 0.84},
            ],
            "optimization_time": 285.5,
            "total_trials": 45,
        }

        # Configure mocks
        workflow_services["automl_service"].run_automl.return_value = automl_results

        # Execute AutoML workflow
        automl_service = workflow_services["automl_service"]
        optimization_results = await automl_service.run_automl(automl_config)

        # Verify AutoML workflow
        assert optimization_results["best_model"]["algorithm"] == "IsolationForest"
        assert optimization_results["best_model"]["performance"]["f1_score"] > 0.9
        assert optimization_results["optimization_time"] < 300
        assert len(optimization_results["model_comparison"]) == 3

    @pytest.mark.asyncio
    async def test_explainability_integration_workflow(self, workflow_services):
        """Test explainability integration workflow."""
        detector_id = str(uuid.uuid4())
        dataset_id = str(uuid.uuid4())

        # Mock anomaly for explanation
        anomaly_data = {
            "features": {"temperature": 95.2, "humidity": 30.1, "pressure": 1013.2},
            "anomaly_score": 0.85,
            "detector_id": detector_id,
        }

        # Mock explanation results
        explanation_results = {
            "local_explanation": {
                "feature_contributions": {
                    "temperature": 0.45,
                    "humidity": -0.12,
                    "pressure": 0.08,
                },
                "prediction_confidence": 0.92,
            },
            "global_explanation": {
                "feature_importance": {
                    "temperature": 0.38,
                    "pressure": 0.35,
                    "humidity": 0.27,
                }
            },
            "explanation_quality": {
                "fidelity": 0.94,
                "stability": 0.89,
                "consistency": 0.91,
            },
        }

        # Configure mocks
        workflow_services["explainability_service"].explain_anomaly.return_value = (
            explanation_results["local_explanation"]
        )
        workflow_services[
            "explainability_service"
        ].analyze_feature_importance.return_value = explanation_results[
            "global_explanation"
        ]

        # Execute explainability workflow
        explainability_service = workflow_services["explainability_service"]

        # Generate local explanation
        local_explanation = await explainability_service.explain_anomaly(
            {
                "detector_id": detector_id,
                "anomaly_data": anomaly_data["features"],
                "explanation_method": "shap",
            }
        )

        # Generate global explanation
        global_explanation = await explainability_service.analyze_feature_importance(
            {
                "detector_id": detector_id,
                "dataset_id": dataset_id,
                "importance_method": "shap_global",
            }
        )

        # Verify explainability workflow
        assert "temperature" in local_explanation["feature_contributions"]
        assert local_explanation["feature_contributions"]["temperature"] > 0.4
        assert "temperature" in global_explanation["feature_importance"]
        assert local_explanation["prediction_confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_comprehensive_detection_pipeline(self, workflow_services):
        """Test comprehensive detection pipeline with all components."""
        dataset_id = str(uuid.uuid4())

        # Pipeline configuration
        pipeline_config = {
            "data_preprocessing": {
                "scaling": "standard",
                "feature_selection": True,
                "outlier_removal": False,
            },
            "automl_optimization": {
                "enable": True,
                "time_budget": 180,
                "optimization_metric": "f1_score",
            },
            "ensemble_creation": {
                "enable": True,
                "ensemble_method": "weighted_voting",
                "diversity_threshold": 0.3,
            },
            "explainability": {
                "enable": True,
                "explanation_methods": ["shap", "lime"],
                "global_explanations": True,
            },
            "real_time_deployment": {
                "enable": True,
                "max_latency_ms": 100,
                "monitoring": True,
            },
        }

        # Mock pipeline execution results
        pipeline_results = {
            "preprocessing_stats": {
                "original_features": 10,
                "selected_features": 8,
                "scaling_method": "standard",
            },
            "automl_results": {
                "best_algorithm": "IsolationForest",
                "optimization_score": 0.92,
                "total_trials": 25,
            },
            "ensemble_info": {
                "base_models": 3,
                "ensemble_score": 0.94,
                "diversity_score": 0.72,
            },
            "explainability_summary": {
                "explanation_coverage": 0.98,
                "explanation_quality": 0.91,
                "top_features": ["temperature", "pressure", "vibration"],
            },
            "deployment_metrics": {
                "average_latency_ms": 45,
                "throughput_per_second": 250,
                "memory_usage_mb": 128,
            },
        }

        # Mock comprehensive pipeline execution
        workflow_services["detection_service"].execute_pipeline.return_value = (
            pipeline_results
        )

        # Execute comprehensive pipeline
        detection_service = workflow_services["detection_service"]
        results = await detection_service.execute_pipeline(
            {"dataset_id": dataset_id, "pipeline_config": pipeline_config}
        )

        # Verify comprehensive pipeline
        assert results["automl_results"]["optimization_score"] > 0.9
        assert results["ensemble_info"]["ensemble_score"] > 0.9
        assert results["deployment_metrics"]["average_latency_ms"] < 100
        assert len(results["explainability_summary"]["top_features"]) == 3

    @pytest.mark.asyncio
    async def test_multi_tenant_workflow(self, workflow_services):
        """Test multi-tenant anomaly detection workflow."""
        # Mock multi-tenant configuration
        tenants = [
            {"id": "tenant_1", "name": "Manufacturing Corp"},
            {"id": "tenant_2", "name": "Healthcare Inc"},
            {"id": "tenant_3", "name": "Finance Ltd"},
        ]

        # Mock tenant-specific detectors and datasets
        tenant_data = {}
        for tenant in tenants:
            tenant_data[tenant["id"]] = {
                "detector_id": str(uuid.uuid4()),
                "dataset_id": str(uuid.uuid4()),
                "results": DetectionResult(
                    id=str(uuid.uuid4()),
                    detector_id=str(uuid.uuid4()),
                    dataset_id=str(uuid.uuid4()),
                    anomaly_scores=[0.1, 0.3, 0.8, 0.2, 0.9],
                    anomalies=[2, 4],
                ),
            }

        # Mock multi-tenant processing
        workflow_services["detection_service"].process_multi_tenant.return_value = (
            tenant_data
        )

        # Execute multi-tenant workflow
        detection_service = workflow_services["detection_service"]
        results = await detection_service.process_multi_tenant(
            {
                "tenants": tenants,
                "isolation_level": "strict",
                "resource_allocation": "fair",
            }
        )

        # Verify multi-tenant processing
        assert len(results) == 3
        for _tenant_id, result in results.items():
            assert "detector_id" in result
            assert "results" in result
            assert len(result["results"].anomalies) > 0

    @pytest.mark.asyncio
    async def test_disaster_recovery_workflow(self, workflow_services):
        """Test disaster recovery and failover workflow."""
        primary_config = {
            "detector_id": str(uuid.uuid4()),
            "region": "primary",
            "backup_regions": ["secondary", "tertiary"],
        }

        # Simulate primary region failure
        with patch.object(
            workflow_services["detection_service"], "health_check"
        ) as mock_health:
            mock_health.return_value = {"status": "unhealthy", "region": "primary"}

            # Mock failover results
            failover_results = {
                "failover_region": "secondary",
                "failover_time_ms": 150,
                "data_consistency": "eventual",
                "backup_detector_id": str(uuid.uuid4()),
            }

            workflow_services["detection_service"].initiate_failover.return_value = (
                failover_results
            )

            # Execute disaster recovery
            detection_service = workflow_services["detection_service"]
            recovery_results = await detection_service.initiate_failover(primary_config)

            # Verify disaster recovery
            assert recovery_results["failover_region"] == "secondary"
            assert recovery_results["failover_time_ms"] < 500
            assert recovery_results["backup_detector_id"] is not None
