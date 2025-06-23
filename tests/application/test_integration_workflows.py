"""Integration workflow tests for application layer - Phase 1 Coverage Enhancement."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import UUID, uuid4
import tempfile

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ModelPersistenceService,
    ExperimentTrackingService,
)
from pynomaly.application.use_cases.detect_anomalies import (
    DetectAnomaliesUseCase,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse
)
from pynomaly.application.use_cases.train_detector import (
    TrainDetectorUseCase,
    TrainDetectorRequest,
    TrainDetectorResponse
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore, ConfidenceInterval
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator, EnsembleAggregator, FeatureValidator
from pynomaly.infrastructure.repositories import (
    InMemoryDetectorRepository,
    InMemoryResultRepository,
    InMemoryDatasetRepository,
)


class TestApplicationLayerIntegration:
    """Integration tests combining multiple application layer components."""
    
    @pytest.fixture
    def complete_application_stack(self):
        """Create a complete application stack for integration testing."""
        # Repositories
        detector_repo = InMemoryDetectorRepository()
        result_repo = InMemoryResultRepository()
        dataset_repo = InMemoryDatasetRepository()
        
        # Domain services
        scorer = AnomalyScorer()
        threshold_calc = ThresholdCalculator()
        aggregator = EnsembleAggregator()
        feature_validator = FeatureValidator()
        
        # Application services
        detection_service = DetectionService(
            detector_repository=detector_repo,
            result_repository=result_repo,
            anomaly_scorer=scorer,
            threshold_calculator=threshold_calc
        )
        
        ensemble_service = EnsembleService(
            detector_repository=detector_repo,
            ensemble_aggregator=aggregator,
            anomaly_scorer=scorer
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_path = Path(tmp_dir)
            
            persistence_service = ModelPersistenceService(
                detector_repository=detector_repo,
                storage_path=storage_path
            )
            
            tracking_service = ExperimentTrackingService(
                tracking_path=storage_path / "experiments"
            )
            
            # Use cases
            detect_use_case = DetectAnomaliesUseCase(
                detector_repository=detector_repo,
                feature_validator=feature_validator
            )
            
            train_use_case = TrainDetectorUseCase(
                detector_repository=detector_repo,
                feature_validator=feature_validator
            )
            
            yield {
                "repos": {
                    "detector": detector_repo,
                    "result": result_repo,
                    "dataset": dataset_repo
                },
                "services": {
                    "detection": detection_service,
                    "ensemble": ensemble_service,
                    "persistence": persistence_service,
                    "tracking": tracking_service
                },
                "use_cases": {
                    "detect": detect_use_case,
                    "train": train_use_case
                },
                "domain_services": {
                    "scorer": scorer,
                    "threshold_calc": threshold_calc,
                    "aggregator": aggregator,
                    "validator": feature_validator
                }
            }
    
    @pytest.mark.asyncio
    async def test_complete_training_and_detection_workflow(self, complete_application_stack):
        """Test complete workflow from training to detection."""
        stack = complete_application_stack
        
        # Step 1: Create and train detector
        detector = Detector(
            name="workflow_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
            hyperparameters={"n_estimators": 100, "random_state": 42}
        )
        await stack["repos"]["detector"].save(detector)
        
        # Step 2: Prepare training data
        training_features = np.random.RandomState(42).normal(0, 1, (500, 8))
        training_dataset = Dataset(name="training_data", features=training_features)
        
        # Mock training process
        with patch.object(detector, 'fit') as mock_fit:
            with patch.object(stack["domain_services"]["validator"], 'validate_numeric_features') as mock_validate:
                with patch.object(stack["domain_services"]["validator"], 'check_data_quality') as mock_quality:
                    mock_validate.return_value = [f"feature_{i}" for i in range(8)]\n                    mock_quality.return_value = {"quality_score": 0.9, "missing_values": [], "constant_features": []}\n                    \n                    train_request = TrainDetectorRequest(\n                        detector_id=detector.id,\n                        dataset=training_dataset,\n                        validate_data=True,\n                        save_model=True\n                    )\n                    \n                    train_response = await stack["use_cases"]["train"].execute(train_request)\n                    \n                    assert isinstance(train_response, TrainDetectorResponse)\n                    assert train_response.detector_id == detector.id\n                    assert train_response.training_time_ms > 0\n                    mock_fit.assert_called_once()\n        \n        # Step 3: Save trained model\n        detector._is_fitted = True\n        with patch('pickle.dump') as mock_dump:\n            model_path = await stack["services"]["persistence"].save_model(\n                detector_id=detector.id,\n                metadata={"training_dataset": "training_data", "version": "1.0"}\n            )\n            assert model_path is not None\n            mock_dump.assert_called_once()\n        \n        # Step 4: Perform detection on new data\n        test_features = np.random.RandomState(123).normal(0, 1, (200, 8))\n        test_dataset = Dataset(name="test_data", features=test_features)\n        \n        # Mock detection process\n        mock_scores = [AnomalyScore(v) for v in np.random.RandomState(123).beta(2, 8, 200)]\n        mock_result = DetectionResult(\n            detector_id=detector.id,\n            dataset_id=test_dataset.id,\n            anomalies=[],\n            scores=mock_scores,\n            labels=np.array([0] * 200),\n            threshold=0.7\n        )\n        \n        with patch.object(detector, 'detect', return_value=mock_result):\n            with patch.object(stack["domain_services"]["validator"], 'check_data_quality') as mock_quality:\n                mock_quality.return_value = {"quality_score": 0.85}\n                \n                detect_request = DetectAnomaliesRequest(\n                    detector_id=detector.id,\n                    dataset=test_dataset,\n                    validate_features=True,\n                    save_results=True\n                )\n                \n                detect_response = await stack["use_cases"]["detect"].execute(detect_request)\n                \n                assert isinstance(detect_response, DetectAnomaliesResponse)\n                assert detect_response.result.detector_id == detector.id\n                assert len(detect_response.result.scores) == 200\n    \n    @pytest.mark.asyncio\n    async def test_ensemble_workflow_with_tracking(self, complete_application_stack):\n        """Test ensemble creation and evaluation with experiment tracking."""\n        stack = complete_application_stack\n        \n        # Step 1: Create multiple detectors\n        detectors = []\n        algorithms = ["isolation_forest", "local_outlier_factor", "one_class_svm"]\n        \n        for i, algo in enumerate(algorithms):\n            detector = Detector(\n                name=f"detector_{i}_{algo}",\n                algorithm=algo,\n                contamination=ContaminationRate(0.1 + i * 0.01)\n            )\n            detector._is_fitted = True\n            await stack["repos"]["detector"].save(detector)\n            detectors.append(detector)\n        \n        # Step 2: Start experiment tracking\n        experiment_id = await stack["services"]["tracking"].create_experiment(\n            name="Ensemble Comparison Experiment",\n            description="Comparing individual vs ensemble performance",\n            tags=["ensemble", "comparison", "production"]\n        )\n        \n        # Step 3: Create test dataset\n        features = np.random.RandomState(42).normal(0, 1, (300, 6))\n        targets = np.random.RandomState(42).choice([0, 1], size=300, p=[0.92, 0.08])\n        test_dataset = Dataset(name="ensemble_test_data", features=features, targets=targets)\n        \n        # Step 4: Evaluate individual detectors\n        individual_results = {}\n        for detector in detectors:\n            # Mock individual detection\n            mock_scores = [AnomalyScore(v) for v in np.random.RandomState(42).random(300)]\n            mock_labels = np.random.RandomState(42).choice([0, 1], size=300, p=[0.9, 0.1])\n            \n            with patch.object(detector, 'detect') as mock_detect:\n                mock_result = DetectionResult(\n                    detector_id=detector.id,\n                    dataset_id=test_dataset.id,\n                    anomalies=[],\n                    scores=mock_scores,\n                    labels=mock_labels,\n                    threshold=0.7\n                )\n                mock_detect.return_value = mock_result\n                \n                result = await stack["services"]["detection"]._detect_async(detector, test_dataset)\n                individual_results[detector.id] = result\n                \n                # Log individual performance\n                from sklearn.metrics import f1_score, precision_score, recall_score\n                f1 = f1_score(targets, mock_labels)\n                precision = precision_score(targets, mock_labels)\n                recall = recall_score(targets, mock_labels)\n                \n                run_id = await stack["services"]["tracking"].log_run(\n                    experiment_id=experiment_id,\n                    detector_name=detector.name,\n                    dataset_name=test_dataset.name,\n                    parameters=detector.hyperparameters,\n                    metrics={"f1": f1, "precision": precision, "recall": recall}\n                )\n        \n        # Step 5: Create and evaluate ensemble\n        detector_ids = [d.id for d in detectors]\n        \n        with patch.object(stack["services"]["ensemble"], 'create_ensemble') as mock_create_ensemble:\n            mock_ensemble = Mock()\n            mock_ensemble.name = "Test Ensemble"\n            mock_ensemble.base_detectors = detectors\n            mock_create_ensemble.return_value = mock_ensemble\n            \n            # Mock ensemble detection\n            ensemble_scores = [AnomalyScore(v) for v in np.random.RandomState(42).beta(1.5, 6, 300)]\n            ensemble_labels = np.random.RandomState(42).choice([0, 1], size=300, p=[0.88, 0.12])\n            \n            with patch.object(mock_ensemble, 'detect') as mock_ensemble_detect:\n                ensemble_result = DetectionResult(\n                    detector_id=uuid4(),\n                    dataset_id=test_dataset.id,\n                    anomalies=[],\n                    scores=ensemble_scores,\n                    labels=ensemble_labels,\n                    threshold=0.75\n                )\n                mock_ensemble_detect.return_value = ensemble_result\n                \n                ensemble = await stack["services"]["ensemble"].create_ensemble(\n                    name="Test Ensemble",\n                    detector_ids=detector_ids,\n                    aggregation_method="average"\n                )\n                \n                result = ensemble.detect(test_dataset)\n                \n                # Log ensemble performance\n                ensemble_f1 = f1_score(targets, ensemble_labels)\n                ensemble_precision = precision_score(targets, ensemble_labels)\n                ensemble_recall = recall_score(targets, ensemble_labels)\n                \n                ensemble_run_id = await stack["services"]["tracking"].log_run(\n                    experiment_id=experiment_id,\n                    detector_name="Ensemble",\n                    dataset_name=test_dataset.name,\n                    parameters={"aggregation": "average", "n_detectors": len(detectors)},\n                    metrics={"f1": ensemble_f1, "precision": ensemble_precision, "recall": ensemble_recall}\n                )\n        \n        # Step 6: Compare results\n        comparison_df = await stack["services"]["tracking"].compare_runs(\n            experiment_id=experiment_id,\n            metric="f1"\n        )\n        \n        assert len(comparison_df) == 4  # 3 individual + 1 ensemble\n        assert "f1" in comparison_df.columns\n        assert "detector" in comparison_df.columns\n        \n        # Step 7: Get best performing approach\n        best_run = await stack["services"]["tracking"].get_best_run(\n            experiment_id=experiment_id,\n            metric="f1"\n        )\n        \n        assert best_run is not None\n        assert "f1" in best_run["metrics"]\n    \n    @pytest.mark.asyncio\n    async def test_model_versioning_and_rollback_workflow(self, complete_application_stack):\n        """Test model versioning and rollback workflow."""\n        stack = complete_application_stack\n        \n        # Step 1: Create initial detector\n        detector = Detector(\n            name="versioned_detector",\n            algorithm="isolation_forest",\n            contamination=ContaminationRate(0.1)\n        )\n        detector._is_fitted = True\n        await stack["repos"]["detector"].save(detector)\n        \n        # Step 2: Save initial model version\n        with patch('pickle.dump') as mock_dump:\n            v1_path = await stack["services"]["persistence"].save_model(\n                detector_id=detector.id,\n                metadata={"version": "1.0", "performance": {"f1": 0.85}}\n            )\n            assert "model.pkl" in v1_path\n        \n        # Step 3: Update detector and save new version\n        detector.hyperparameters["n_estimators"] = 200\n        detector.contamination_rate = ContaminationRate(0.08)\n        \n        with patch('pickle.dump') as mock_dump:\n            v2_path = await stack["services"]["persistence"].save_model(\n                detector_id=detector.id,\n                metadata={"version": "2.0", "performance": {"f1": 0.87}}\n            )\n        \n        # Step 4: List saved models to verify versions\n        saved_models = await stack["services"]["persistence"].list_saved_models()\n        \n        detector_models = saved_models.get(str(detector.id))\n        assert detector_models is not None\n        assert "version" in detector_models\n        \n        # Step 5: Test model loading (simulating rollback)\n        with patch('pickle.load') as mock_load:\n            mock_load.return_value = detector\n            \n            loaded_detector = await stack["services"]["persistence"].load_model(\n                detector_id=detector.id\n            )\n            \n            assert loaded_detector is not None\n            mock_load.assert_called_once()\n    \n    @pytest.mark.asyncio\n    async def test_parallel_detection_with_performance_monitoring(self, complete_application_stack):\n        """Test parallel detection across multiple detectors with performance monitoring."""\n        stack = complete_application_stack\n        \n        # Step 1: Create multiple detectors for parallel execution\n        detectors = []\n        for i in range(5):\n            detector = Detector(\n                name=f"parallel_detector_{i}",\n                algorithm=f"algorithm_{i}",\n                contamination=ContaminationRate(0.05 + i * 0.01)\n            )\n            detector._is_fitted = True\n            await stack["repos"]["detector"].save(detector)\n            detectors.append(detector)\n        \n        # Step 2: Create large dataset for performance testing\n        features = np.random.RandomState(42).normal(0, 1, (1000, 10))\n        large_dataset = Dataset(name="performance_test_data", features=features)\n        \n        # Step 3: Mock parallel detection\n        detector_ids = [d.id for d in detectors]\n        \n        async def mock_detect_async(detector, dataset):\n            \"\"\"Mock async detection with simulated processing time.\"\"\"\n            await asyncio.sleep(0.01)  # Simulate processing time\n            scores = [AnomalyScore(v) for v in np.random.random(len(dataset.features))]\n            labels = np.random.choice([0, 1], size=len(dataset.features), p=[0.95, 0.05])\n            \n            return DetectionResult(\n                detector_id=detector.id,\n                dataset_id=dataset.id,\n                anomalies=[],\n                scores=scores,\n                labels=labels,\n                threshold=0.8\n            )\n        \n        with patch.object(stack["services"]["detection"], '_detect_async', side_effect=mock_detect_async):\n            start_time = datetime.now()\n            \n            # Execute parallel detection\n            results = await stack["services"]["detection"].detect_with_multiple_detectors(\n                detector_ids=detector_ids,\n                dataset=large_dataset,\n                save_results=True\n            )\n            \n            end_time = datetime.now()\n            processing_time = (end_time - start_time).total_seconds()\n            \n            # Verify results\n            assert len(results) == 5\n            assert all(isinstance(result, DetectionResult) for result in results.values())\n            assert all(detector_id in results for detector_id in detector_ids)\n            \n            # Performance should be better than sequential (though we can't test real parallelism in sync tests)\n            assert processing_time < 1.0  # Should complete quickly with mocking\n    \n    @pytest.mark.asyncio\n    async def test_comprehensive_error_handling_workflow(self, complete_application_stack):\n        """Test comprehensive error handling across the application stack."""\n        stack = complete_application_stack\n        \n        # Test 1: Training with insufficient data\n        detector = Detector(name="error_test", algorithm="test", contamination=ContaminationRate(0.1))\n        await stack["repos"]["detector"].save(detector)\n        \n        small_features = np.random.random((3, 2))  # Too small\n        small_dataset = Dataset(name="insufficient_data", features=small_features)\n        \n        train_request = TrainDetectorRequest(detector_id=detector.id, dataset=small_dataset)\n        \n        from pynomaly.domain.exceptions import InsufficientDataError\n        with pytest.raises(InsufficientDataError):\n            await stack["use_cases"]["train"].execute(train_request)\n        \n        # Test 2: Detection with unfitted detector\n        test_features = np.random.random((50, 2))\n        test_dataset = Dataset(name="test_data", features=test_features)\n        \n        detect_request = DetectAnomaliesRequest(detector_id=detector.id, dataset=test_dataset)\n        \n        from pynomaly.domain.exceptions import DetectorNotFittedError\n        with pytest.raises(DetectorNotFittedError):\n            await stack["use_cases"]["detect"].execute(detect_request)\n        \n        # Test 3: Model persistence with non-existent detector\n        non_existent_id = uuid4()\n        \n        with pytest.raises(ValueError, match="Detector .* not found"):\n            await stack["services"]["persistence"].save_model(detector_id=non_existent_id)\n        \n        # Test 4: Experiment tracking with invalid experiment\n        with pytest.raises(ValueError, match="Experiment .* not found"):\n            await stack["services"]["tracking"].log_run(\n                experiment_id="non_existent_experiment",\n                detector_name="test",\n                dataset_name="test",\n                parameters={},\n                metrics={}\n            )\n        \n        # Test 5: Ensemble creation with non-existent detectors\n        with pytest.raises(ValueError, match="Detector .* not found"):\n            await stack["services"]["ensemble"].create_ensemble(\n                name="Invalid Ensemble",\n                detector_ids=[uuid4(), uuid4()]\n            )