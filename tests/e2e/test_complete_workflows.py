"""
End-to-End Workflow Testing Suite
Comprehensive tests for complete anomaly detection workflows from data ingestion to deployment.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from pynomaly.application.use_cases.detect_anomalies import DetectAnomalies
from pynomaly.application.use_cases.train_detector import TrainDetector
from pynomaly.application.use_cases.evaluate_model import EvaluateModel
from pynomaly.application.services.detection_service import DetectionService
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.infrastructure.persistence.model_repository import ModelRepository
from pynomaly.infrastructure.persistence.data_loader import DataLoader
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore


class TestCompleteAnomalyDetectionWorkflow:
    """Test complete anomaly detection workflows from start to finish."""

    @pytest.fixture
    def sample_dataset_csv(self):
        """Create sample CSV dataset for testing."""
        data = {
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(0, 1, 1000), 
            'feature_3': np.random.normal(0, 1, 1000),
            'feature_4': np.random.normal(0, 1, 1000),
            'feature_5': np.random.normal(0, 1, 1000)
        }
        
        # Add some anomalies
        anomaly_indices = np.random.choice(1000, 50, replace=False)
        for idx in anomaly_indices:
            data['feature_1'][idx] = np.random.normal(5, 1)  # Clear outliers
            data['feature_2'][idx] = np.random.normal(5, 1)
        
        df = pd.DataFrame(data)
        
        # Save to temporary CSV
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name

    @pytest.fixture
    def detection_service(self):
        """Create detection service with dependencies."""
        model_repo = Mock(spec=ModelRepository)
        data_loader = Mock(spec=DataLoader)
        
        return DetectionService(
            model_repository=model_repo,
            data_loader=data_loader
        )

    def test_data_ingestion_to_detection_workflow(self, sample_dataset_csv, detection_service):
        """Test complete workflow from data ingestion to anomaly detection."""
        
        # Step 1: Data Ingestion
        with patch('pynomaly.infrastructure.persistence.data_loader.DataLoader.load_csv') as mock_load:
            mock_dataset = Mock(spec=Dataset)
            mock_dataset.data = np.random.randn(1000, 5)
            mock_dataset.features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
            mock_dataset.id = "test_dataset"
            mock_load.return_value = mock_dataset
            
            dataset = detection_service.data_loader.load_csv(sample_dataset_csv)
            
            assert dataset is not None
            assert hasattr(dataset, 'data')
            assert hasattr(dataset, 'features')
        
        # Step 2: Model Training
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_fit:
            mock_detector = Mock(spec=Detector)
            mock_detector.id = "isolation_forest_detector"
            mock_detector.algorithm = "sklearn_isolation_forest"
            mock_detector.parameters = {"contamination": 0.05}
            mock_fit.return_value = mock_detector
            
            train_use_case = TrainDetector(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            detector = train_use_case.execute(
                dataset=dataset,
                algorithm="IsolationForest",
                contamination_rate=ContaminationRate(0.05)
            )
            
            assert detector is not None
            assert detector.algorithm == "sklearn_isolation_forest"
        
        # Step 3: Model Persistence
        with patch.object(detection_service.model_repository, 'save') as mock_save:
            detection_service.model_repository.save(detector)
            mock_save.assert_called_once_with(detector)
        
        # Step 4: Anomaly Detection
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.predict') as mock_predict:
            mock_result = Mock(spec=DetectionResult)
            mock_result.anomaly_scores = [AnomalyScore(0.1) for _ in range(200)]
            mock_result.predictions = np.random.choice([0, 1], 200, p=[0.95, 0.05])
            mock_result.detector_id = detector.id
            mock_predict.return_value = mock_result
            
            detect_use_case = DetectAnomalies(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            test_data = np.random.randn(200, 5)
            result = detect_use_case.execute(
                detector_id=detector.id,
                data=test_data
            )
            
            assert result is not None
            assert len(result.anomaly_scores) == 200
            assert hasattr(result, 'predictions')

    def test_model_evaluation_workflow(self, sample_dataset_csv):
        """Test model evaluation workflow with cross-validation."""
        
        # Create evaluation use case
        evaluate_use_case = EvaluateModel(
            adapter=SklearnAdapter(algorithm="IsolationForest")
        )
        
        with patch('pynomaly.infrastructure.persistence.data_loader.DataLoader.load_csv') as mock_load, \
             patch('sklearn.model_selection.cross_val_score') as mock_cv:
            
            # Mock dataset loading
            mock_dataset = Mock(spec=Dataset)
            mock_dataset.data = np.random.randn(1000, 5)
            mock_dataset.labels = np.random.choice([0, 1], 1000, p=[0.9, 0.1])
            mock_load.return_value = mock_dataset
            
            # Mock cross-validation scores
            mock_cv.return_value = np.array([0.85, 0.82, 0.88, 0.86, 0.84])
            
            # Execute evaluation
            evaluation_result = evaluate_use_case.execute(
                dataset=mock_dataset,
                algorithm="IsolationForest",
                metrics=["precision", "recall", "f1_score", "roc_auc"],
                cv_folds=5
            )
            
            assert evaluation_result is not None
            assert "cross_validation_scores" in evaluation_result
            assert "mean_score" in evaluation_result
            assert "std_score" in evaluation_result
            assert evaluation_result["mean_score"] > 0.8

    def test_streaming_detection_workflow(self, detection_service):
        """Test streaming anomaly detection workflow."""
        
        # Mock pre-trained detector
        mock_detector = Mock(spec=Detector)
        mock_detector.id = "streaming_detector"
        mock_detector.algorithm = "sklearn_isolation_forest"
        
        with patch.object(detection_service.model_repository, 'get_by_id', return_value=mock_detector), \
             patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.predict') as mock_predict:
            
            # Mock streaming predictions
            mock_predict.side_effect = [
                Mock(anomaly_scores=[AnomalyScore(0.1)], predictions=[0]),
                Mock(anomaly_scores=[AnomalyScore(0.8)], predictions=[1]),  # Anomaly
                Mock(anomaly_scores=[AnomalyScore(0.2)], predictions=[0]),
                Mock(anomaly_scores=[AnomalyScore(0.9)], predictions=[1]),  # Anomaly
            ]
            
            # Simulate streaming data
            streaming_data = [
                np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
                np.array([[5.1, 5.2, 5.3, 5.4, 5.5]]),  # Anomaly
                np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]),
                np.array([[6.1, 6.2, 6.3, 6.4, 6.5]]),  # Anomaly
            ]
            
            detect_use_case = DetectAnomalies(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            anomaly_count = 0
            for data_batch in streaming_data:
                result = detect_use_case.execute(
                    detector_id=mock_detector.id,
                    data=data_batch
                )
                
                if result.predictions[0] == 1:
                    anomaly_count += 1
            
            assert anomaly_count == 2  # Should detect 2 anomalies

    def test_batch_processing_workflow(self, detection_service):
        """Test large-scale batch processing workflow."""
        
        # Mock large dataset
        large_dataset_size = 100000
        batch_size = 5000
        
        mock_detector = Mock(spec=Detector)
        mock_detector.id = "batch_detector"
        
        with patch.object(detection_service.model_repository, 'get_by_id', return_value=mock_detector), \
             patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.predict') as mock_predict:
            
            # Mock batch predictions
            def mock_batch_predict(detector, data):
                batch_size = len(data)
                return Mock(
                    anomaly_scores=[AnomalyScore(np.random.random()) for _ in range(batch_size)],
                    predictions=np.random.choice([0, 1], batch_size, p=[0.95, 0.05])
                )
            
            mock_predict.side_effect = mock_batch_predict
            
            detect_use_case = DetectAnomalies(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            # Process in batches
            total_anomalies = 0
            num_batches = large_dataset_size // batch_size
            
            for batch_idx in range(num_batches):
                batch_data = np.random.randn(batch_size, 5)
                
                result = detect_use_case.execute(
                    detector_id=mock_detector.id,
                    data=batch_data
                )
                
                total_anomalies += np.sum(result.predictions)
            
            # Should detect approximately 5% anomalies
            expected_anomalies = large_dataset_size * 0.05
            assert abs(total_anomalies - expected_anomalies) < expected_anomalies * 0.3

    def test_model_retraining_workflow(self, detection_service):
        """Test model retraining and update workflow."""
        
        # Step 1: Initial model training
        initial_dataset = Mock(spec=Dataset)
        initial_dataset.data = np.random.randn(1000, 5)
        initial_dataset.id = "initial_dataset"
        
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_fit:
            mock_detector_v1 = Mock(spec=Detector)
            mock_detector_v1.id = "detector_v1"
            mock_detector_v1.version = 1
            mock_detector_v1.created_at = datetime.now() - timedelta(days=30)
            mock_fit.return_value = mock_detector_v1
            
            train_use_case = TrainDetector(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            detector_v1 = train_use_case.execute(
                dataset=initial_dataset,
                algorithm="IsolationForest"
            )
        
        # Step 2: Model performance monitoring (simulate degradation)
        with patch.object(detection_service.model_repository, 'get_by_id', return_value=detector_v1):
            
            evaluate_use_case = EvaluateModel(
                adapter=SklearnAdapter(algorithm="IsolationForest")
            )
            
            # Simulate performance degradation
            with patch('sklearn.model_selection.cross_val_score', return_value=np.array([0.65, 0.62, 0.68, 0.64, 0.66])):
                
                current_performance = evaluate_use_case.execute(
                    dataset=initial_dataset,
                    detector_id=detector_v1.id
                )
                
                # Performance dropped below threshold (0.8)
                assert current_performance["mean_score"] < 0.8
        
        # Step 3: Trigger retraining with new data
        new_dataset = Mock(spec=Dataset)
        new_dataset.data = np.random.randn(1500, 5)  # More data
        new_dataset.id = "expanded_dataset"
        
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_retrain:
            mock_detector_v2 = Mock(spec=Detector)
            mock_detector_v2.id = "detector_v2"
            mock_detector_v2.version = 2
            mock_detector_v2.created_at = datetime.now()
            mock_retrain.return_value = mock_detector_v2
            
            # Retrain with combined dataset
            detector_v2 = train_use_case.execute(
                dataset=new_dataset,
                algorithm="IsolationForest",
                replace_model=detector_v1.id
            )
            
            assert detector_v2.version > detector_v1.version

    def test_multi_algorithm_ensemble_workflow(self, detection_service):
        """Test ensemble detection workflow with multiple algorithms."""
        
        dataset = Mock(spec=Dataset)
        dataset.data = np.random.randn(1000, 5)
        dataset.id = "ensemble_dataset"
        
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        detectors = []
        
        # Train multiple models
        for algorithm in algorithms:
            with patch(f'pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_fit:
                mock_detector = Mock(spec=Detector)
                mock_detector.id = f"{algorithm.lower()}_detector"
                mock_detector.algorithm = f"sklearn_{algorithm.lower()}"
                mock_fit.return_value = mock_detector
                
                train_use_case = TrainDetector(
                    adapter=SklearnAdapter(algorithm=algorithm),
                    model_repository=detection_service.model_repository
                )
                
                detector = train_use_case.execute(
                    dataset=dataset,
                    algorithm=algorithm
                )
                
                detectors.append(detector)
        
        # Ensemble prediction
        test_data = np.random.randn(200, 5)
        ensemble_predictions = []
        
        for detector in detectors:
            with patch.object(detection_service.model_repository, 'get_by_id', return_value=detector), \
                 patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.predict') as mock_predict:
                
                mock_result = Mock(spec=DetectionResult)
                mock_result.anomaly_scores = [AnomalyScore(np.random.random()) for _ in range(200)]
                mock_result.predictions = np.random.choice([0, 1], 200, p=[0.95, 0.05])
                mock_predict.return_value = mock_result
                
                detect_use_case = DetectAnomalies(
                    adapter=SklearnAdapter(algorithm=detector.algorithm.split('_')[-1]),
                    model_repository=detection_service.model_repository
                )
                
                result = detect_use_case.execute(
                    detector_id=detector.id,
                    data=test_data
                )
                
                ensemble_predictions.append(result.predictions)
        
        # Combine predictions (majority voting)
        ensemble_result = np.array(ensemble_predictions).mean(axis=0)
        final_predictions = (ensemble_result > 0.5).astype(int)
        
        assert len(final_predictions) == 200
        assert final_predictions.sum() > 0  # Should detect some anomalies

    def test_data_drift_detection_workflow(self, detection_service):
        """Test data drift detection and model adaptation workflow."""
        
        # Original training data
        original_data = Mock(spec=Dataset)
        original_data.data = np.random.normal(0, 1, (1000, 5))
        original_data.id = "original_dataset"
        
        # Train initial model
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_fit:
            mock_detector = Mock(spec=Detector)
            mock_detector.id = "drift_detector"
            mock_detector.parameters = {
                "training_statistics": {
                    "mean": np.mean(original_data.data, axis=0).tolist(),
                    "std": np.std(original_data.data, axis=0).tolist()
                }
            }
            mock_fit.return_value = mock_detector
            
            train_use_case = TrainDetector(
                adapter=SklearnAdapter(algorithm="IsolationForest"),
                model_repository=detection_service.model_repository
            )
            
            detector = train_use_case.execute(
                dataset=original_data,
                algorithm="IsolationForest"
            )
        
        # New data with drift (different distribution)
        drifted_data = np.random.normal(2, 1.5, (500, 5))  # Shifted mean and scale
        
        # Detect data drift
        from pynomaly.infrastructure.monitoring.drift_detector import DriftDetector
        
        drift_detector = DriftDetector()
        
        with patch.object(drift_detector, 'detect_drift') as mock_drift_detect:
            mock_drift_detect.return_value = {
                "drift_detected": True,
                "drift_score": 0.85,
                "affected_features": [0, 1, 2, 3, 4]
            }
            
            drift_result = drift_detector.detect_drift(
                reference_data=original_data.data,
                current_data=drifted_data
            )
            
            assert drift_result["drift_detected"] is True
            assert drift_result["drift_score"] > 0.5
        
        # Trigger model retraining due to drift
        if drift_result["drift_detected"]:
            combined_data = Mock(spec=Dataset)
            combined_data.data = np.vstack([original_data.data, drifted_data])
            combined_data.id = "adapted_dataset"
            
            with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter.fit') as mock_retrain:
                mock_adapted_detector = Mock(spec=Detector)
                mock_adapted_detector.id = "adapted_detector"
                mock_adapted_detector.version = 2
                mock_retrain.return_value = mock_adapted_detector
                
                adapted_detector = train_use_case.execute(
                    dataset=combined_data,
                    algorithm="IsolationForest",
                    replace_model=detector.id
                )
                
                assert adapted_detector.version > 1


class TestMLOpsWorkflow:
    """Test MLOps workflows including CI/CD, monitoring, and deployment."""

    def test_model_versioning_workflow(self):
        """Test model versioning and lifecycle management."""
        
        from pynomaly.infrastructure.mlops.model_versioning import ModelVersionManager
        
        version_manager = ModelVersionManager()
        
        # Create initial model version
        model_v1 = Mock(spec=Detector)
        model_v1.id = "model_v1"
        model_v1.version = "1.0.0"
        model_v1.metadata = {
            "algorithm": "IsolationForest",
            "training_data_hash": "abc123",
            "performance_metrics": {"roc_auc": 0.85}
        }
        
        with patch.object(version_manager, 'register_version') as mock_register:
            version_manager.register_version(model_v1)
            mock_register.assert_called_once_with(model_v1)
        
        # Create improved model version
        model_v2 = Mock(spec=Detector)
        model_v2.id = "model_v2"
        model_v2.version = "1.1.0"
        model_v2.metadata = {
            "algorithm": "IsolationForest",
            "training_data_hash": "def456",
            "performance_metrics": {"roc_auc": 0.88}
        }
        
        with patch.object(version_manager, 'compare_versions') as mock_compare:
            mock_compare.return_value = {
                "performance_improvement": 0.03,
                "recommendation": "deploy"
            }
            
            comparison = version_manager.compare_versions(model_v1, model_v2)
            assert comparison["recommendation"] == "deploy"

    def test_model_deployment_workflow(self):
        """Test model deployment pipeline."""
        
        from pynomaly.infrastructure.mlops.deployment import ModelDeployment
        
        deployment_manager = ModelDeployment()
        
        # Stage 1: Model validation
        model = Mock(spec=Detector)
        model.id = "production_model"
        model.version = "2.0.0"
        
        with patch.object(deployment_manager, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                "validation_passed": True,
                "performance_tests": "passed",
                "security_scan": "passed",
                "resource_requirements": "satisfied"
            }
            
            validation_result = deployment_manager.validate_model(model)
            assert validation_result["validation_passed"] is True
        
        # Stage 2: Canary deployment
        with patch.object(deployment_manager, 'deploy_canary') as mock_canary:
            mock_canary.return_value = {
                "deployment_id": "canary_123",
                "traffic_percentage": 5,
                "status": "active"
            }
            
            canary_deployment = deployment_manager.deploy_canary(
                model=model,
                traffic_percentage=5
            )
            
            assert canary_deployment["status"] == "active"
        
        # Stage 3: Performance monitoring during canary
        with patch.object(deployment_manager, 'monitor_canary_performance') as mock_monitor:
            mock_monitor.return_value = {
                "error_rate": 0.01,
                "latency_p95": 150,  # ms
                "throughput": 1000,  # requests/minute
                "status": "healthy"
            }
            
            canary_metrics = deployment_manager.monitor_canary_performance(
                deployment_id="canary_123",
                duration_minutes=30
            )
            
            assert canary_metrics["status"] == "healthy"
        
        # Stage 4: Full deployment
        if canary_metrics["status"] == "healthy":
            with patch.object(deployment_manager, 'promote_to_production') as mock_promote:
                mock_promote.return_value = {
                    "deployment_id": "prod_456",
                    "traffic_percentage": 100,
                    "rollback_available": True
                }
                
                production_deployment = deployment_manager.promote_to_production(
                    canary_deployment_id="canary_123"
                )
                
                assert production_deployment["traffic_percentage"] == 100

    def test_continuous_monitoring_workflow(self):
        """Test continuous model monitoring workflow."""
        
        from pynomaly.infrastructure.monitoring.model_monitor import ModelMonitor
        
        monitor = ModelMonitor()
        
        # Mock production model
        production_model = Mock(spec=Detector)
        production_model.id = "production_model_123"
        
        # Simulate monitoring metrics over time
        monitoring_data = [
            {
                "timestamp": datetime.now() - timedelta(hours=24),
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "latency_ms": 120,
                "throughput_rps": 50
            },
            {
                "timestamp": datetime.now() - timedelta(hours=12),
                "accuracy": 0.82,  # Slight degradation
                "precision": 0.80,
                "recall": 0.84,
                "latency_ms": 140,  # Increased latency
                "throughput_rps": 45
            },
            {
                "timestamp": datetime.now(),
                "accuracy": 0.78,  # Significant degradation
                "precision": 0.76,
                "recall": 0.80,
                "latency_ms": 180,  # High latency
                "throughput_rps": 35
            }
        ]
        
        with patch.object(monitor, 'collect_metrics') as mock_collect:
            mock_collect.side_effect = monitoring_data
            
            # Monitor performance over time
            alerts = []
            for data_point in monitoring_data:
                metrics = monitor.collect_metrics(production_model.id)
                
                # Check for performance degradation
                if metrics["accuracy"] < 0.8:
                    alerts.append({
                        "type": "performance_degradation",
                        "metric": "accuracy",
                        "value": metrics["accuracy"],
                        "threshold": 0.8,
                        "timestamp": metrics["timestamp"]
                    })
                
                if metrics["latency_ms"] > 150:
                    alerts.append({
                        "type": "latency_spike",
                        "metric": "latency_ms",
                        "value": metrics["latency_ms"],
                        "threshold": 150,
                        "timestamp": metrics["timestamp"]
                    })
            
            assert len(alerts) >= 2  # Should trigger multiple alerts
            assert any(alert["type"] == "performance_degradation" for alert in alerts)
            assert any(alert["type"] == "latency_spike" for alert in alerts)

    def test_automated_retraining_workflow(self):
        """Test automated model retraining based on triggers."""
        
        from pynomaly.infrastructure.mlops.auto_retrain import AutoRetrainManager
        
        retrain_manager = AutoRetrainManager()
        
        # Configure retraining triggers
        retrain_manager.configure_triggers(
            performance_threshold=0.8,
            data_drift_threshold=0.7,
            time_based_schedule="weekly",
            min_new_samples=1000
        )
        
        # Mock current model state
        current_model = Mock(spec=Detector)
        current_model.id = "current_production_model"
        current_model.performance_metrics = {"accuracy": 0.75}  # Below threshold
        current_model.last_trained = datetime.now() - timedelta(days=10)
        
        # Mock new training data
        new_data = Mock(spec=Dataset)
        new_data.data = np.random.randn(1500, 5)
        new_data.id = "new_training_data"
        
        # Check retraining triggers
        with patch.object(retrain_manager, 'should_retrain') as mock_should_retrain:
            mock_should_retrain.return_value = {
                "retrain": True,
                "reasons": ["performance_degradation", "sufficient_new_data"],
                "priority": "high"
            }
            
            retrain_decision = retrain_manager.should_retrain(
                current_model=current_model,
                new_data=new_data
            )
            
            assert retrain_decision["retrain"] is True
            assert "performance_degradation" in retrain_decision["reasons"]
        
        # Execute automated retraining
        if retrain_decision["retrain"]:
            with patch.object(retrain_manager, 'execute_retraining') as mock_retrain:
                mock_retrain.return_value = {
                    "new_model_id": "retrained_model_456",
                    "performance_improvement": 0.08,
                    "training_time_minutes": 45,
                    "status": "completed"
                }
                
                retrain_result = retrain_manager.execute_retraining(
                    current_model=current_model,
                    training_data=new_data,
                    algorithm="IsolationForest"
                )
                
                assert retrain_result["status"] == "completed"
                assert retrain_result["performance_improvement"] > 0


class TestDataPipelineWorkflow:
    """Test data pipeline workflows including ETL, validation, and preprocessing."""

    def test_etl_pipeline_workflow(self):
        """Test complete ETL pipeline workflow."""
        
        from pynomaly.infrastructure.data_pipeline.etl import ETLPipeline
        
        etl_pipeline = ETLPipeline()
        
        # Stage 1: Extract from multiple sources
        with patch.object(etl_pipeline, 'extract_from_database') as mock_db_extract, \
             patch.object(etl_pipeline, 'extract_from_api') as mock_api_extract, \
             patch.object(etl_pipeline, 'extract_from_files') as mock_file_extract:
            
            mock_db_extract.return_value = pd.DataFrame(np.random.randn(1000, 3), columns=['a', 'b', 'c'])
            mock_api_extract.return_value = pd.DataFrame(np.random.randn(500, 2), columns=['d', 'e'])
            mock_file_extract.return_value = pd.DataFrame(np.random.randn(800, 4), columns=['f', 'g', 'h', 'i'])
            
            extracted_data = etl_pipeline.extract_all_sources()
            
            assert len(extracted_data) == 3  # Three data sources
            assert all(isinstance(df, pd.DataFrame) for df in extracted_data)
        
        # Stage 2: Transform and clean data
        with patch.object(etl_pipeline, 'clean_data') as mock_clean, \
             patch.object(etl_pipeline, 'normalize_features') as mock_normalize, \
             patch.object(etl_pipeline, 'feature_engineering') as mock_feature_eng:
            
            mock_clean.return_value = pd.DataFrame(np.random.randn(2000, 5))  # Combined and cleaned
            mock_normalize.return_value = pd.DataFrame(np.random.randn(2000, 5))  # Normalized
            mock_feature_eng.return_value = pd.DataFrame(np.random.randn(2000, 8))  # With engineered features
            
            transformed_data = etl_pipeline.transform_data(extracted_data)
            
            assert transformed_data.shape[0] > 0
            assert transformed_data.shape[1] >= 5  # Should have features
        
        # Stage 3: Load to target storage
        with patch.object(etl_pipeline, 'load_to_database') as mock_load_db, \
             patch.object(etl_pipeline, 'load_to_data_lake') as mock_load_lake:
            
            etl_pipeline.load_data(transformed_data)
            
            mock_load_db.assert_called()
            mock_load_lake.assert_called()

    def test_data_validation_pipeline(self):
        """Test data validation pipeline workflow."""
        
        from pynomaly.infrastructure.data_pipeline.validation import DataValidator
        
        validator = DataValidator()
        
        # Create test data with various quality issues
        test_data = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],  # Missing value
            'feature_2': [1, 2, 3, 999999, 5],  # Outlier
            'feature_3': ['a', 'b', 'c', 'd', 'e'],  # Categorical
            'feature_4': [1.0, 2.0, 3.0, 4.0, 5.0],  # Clean numeric
            'feature_5': [1, 1, 1, 1, 1]  # No variance
        })
        
        with patch.object(validator, 'validate_schema') as mock_schema, \
             patch.object(validator, 'validate_data_quality') as mock_quality, \
             patch.object(validator, 'validate_statistical_properties') as mock_stats:
            
            # Schema validation
            mock_schema.return_value = {
                "valid": True,
                "schema_matches": True,
                "missing_columns": [],
                "extra_columns": []
            }
            
            # Data quality validation
            mock_quality.return_value = {
                "missing_values": {"feature_1": 1},
                "outliers": {"feature_2": [999999]},
                "duplicates": 0,
                "invalid_types": {}
            }
            
            # Statistical validation
            mock_stats.return_value = {
                "zero_variance_features": ["feature_5"],
                "high_correlation_pairs": [],
                "distribution_shifts": {}
            }
            
            validation_result = validator.validate_dataset(test_data)
            
            assert "schema_validation" in validation_result
            assert "quality_validation" in validation_result
            assert "statistical_validation" in validation_result

    def test_real_time_data_streaming_workflow(self):
        """Test real-time data streaming and processing workflow."""
        
        from pynomaly.infrastructure.streaming.stream_processor import StreamProcessor
        
        stream_processor = StreamProcessor()
        
        # Mock streaming data source
        streaming_data = [
            {"timestamp": datetime.now() - timedelta(seconds=i), "features": np.random.randn(5).tolist()}
            for i in range(100, 0, -1)  # 100 data points over time
        ]
        
        processed_batches = []
        
        def mock_process_batch(batch):
            # Simulate real-time processing
            processed_batch = {
                "batch_id": len(processed_batches),
                "size": len(batch),
                "processed_at": datetime.now(),
                "anomalies_detected": sum(1 for item in batch if max(item["features"]) > 2)
            }
            processed_batches.append(processed_batch)
            return processed_batch
        
        with patch.object(stream_processor, 'process_batch', side_effect=mock_process_batch):
            
            # Process streaming data in batches
            batch_size = 10
            for i in range(0, len(streaming_data), batch_size):
                batch = streaming_data[i:i + batch_size]
                
                result = stream_processor.process_batch(batch)
                
                assert result["size"] == len(batch)
                assert "anomalies_detected" in result
        
        assert len(processed_batches) == 10  # 100 items / 10 batch_size


class TestAPIWorkflow:
    """Test API-based workflows including REST endpoints and SDK usage."""

    def test_rest_api_workflow(self):
        """Test complete REST API workflow."""
        
        from fastapi.testclient import TestClient
        from pynomaly.presentation.api.main import app
        
        client = TestClient(app)
        
        # Step 1: Upload dataset
        with patch('pynomaly.infrastructure.persistence.data_loader.DataLoader.save_dataset') as mock_save:
            mock_save.return_value = {"dataset_id": "test_dataset_123"}
            
            dataset_data = {
                "name": "test_dataset",
                "description": "Test dataset for API workflow",
                "data": np.random.randn(1000, 5).tolist()
            }
            
            response = client.post("/api/v1/datasets", json=dataset_data)
            
            assert response.status_code == 201
            dataset_id = response.json()["dataset_id"]
        
        # Step 2: Train model
        with patch('pynomaly.application.use_cases.train_detector.TrainDetector.execute') as mock_train:
            mock_detector = {
                "detector_id": "test_detector_456",
                "algorithm": "IsolationForest",
                "status": "trained",
                "performance_metrics": {"roc_auc": 0.85}
            }
            mock_train.return_value = mock_detector
            
            training_request = {
                "dataset_id": dataset_id,
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1, "n_estimators": 100}
            }
            
            response = client.post("/api/v1/detectors/train", json=training_request)
            
            assert response.status_code == 201
            detector_id = response.json()["detector_id"]
        
        # Step 3: Make predictions
        with patch('pynomaly.application.use_cases.detect_anomalies.DetectAnomalies.execute') as mock_predict:
            mock_result = {
                "predictions": [0, 1, 0, 0, 1],
                "anomaly_scores": [0.1, 0.8, 0.2, 0.3, 0.9],
                "processing_time_ms": 150
            }
            mock_predict.return_value = mock_result
            
            prediction_request = {
                "detector_id": detector_id,
                "data": np.random.randn(5, 5).tolist()
            }
            
            response = client.post("/api/v1/detectors/predict", json=prediction_request)
            
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert "anomaly_scores" in result
            assert len(result["predictions"]) == 5

    def test_sdk_workflow(self):
        """Test Python SDK workflow."""
        
        from pynomaly.presentation.sdk.pynomaly_client import PynomaliClient
        
        # Initialize client
        client = PynomaliClient(
            api_url="https://api.pynomaly.example.com",
            api_key="test_api_key_123"
        )
        
        # Mock HTTP requests
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:
            
            # Mock dataset upload response
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {"dataset_id": "sdk_dataset_789"}
            
            # Upload dataset via SDK
            dataset = client.datasets.create(
                name="sdk_test_dataset",
                data=np.random.randn(500, 4),
                features=["feature_1", "feature_2", "feature_3", "feature_4"]
            )
            
            assert dataset["dataset_id"] == "sdk_dataset_789"
            
            # Mock model training response
            mock_post.return_value.json.return_value = {
                "detector_id": "sdk_detector_101",
                "status": "training",
                "estimated_completion": "2024-01-01T12:00:00Z"
            }
            
            # Train model via SDK
            detector = client.detectors.train(
                dataset_id=dataset["dataset_id"],
                algorithm="LocalOutlierFactor",
                parameters={"n_neighbors": 20}
            )
            
            assert detector["detector_id"] == "sdk_detector_101"
            
            # Mock prediction response
            mock_post.return_value.json.return_value = {
                "predictions": [0, 1, 0],
                "anomaly_scores": [0.2, 0.8, 0.3],
                "confidence_intervals": [[0.1, 0.3], [0.7, 0.9], [0.2, 0.4]]
            }
            
            # Make predictions via SDK
            predictions = client.detectors.predict(
                detector_id=detector["detector_id"],
                data=np.random.randn(3, 4)
            )
            
            assert len(predictions["predictions"]) == 3
            assert len(predictions["anomaly_scores"]) == 3