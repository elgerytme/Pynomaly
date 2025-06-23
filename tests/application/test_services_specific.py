"""Specific tests for application service methods to improve coverage - Phase 1."""

from __future__ import annotations

import asyncio
import json
import numpy as np
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
from uuid import UUID, uuid4
import tempfile

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ModelPersistenceService,
    ExperimentTrackingService,
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore, ConfidenceInterval
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator, EnsembleAggregator
from pynomaly.infrastructure.repositories import (
    InMemoryDetectorRepository,
    InMemoryResultRepository,
    InMemoryDatasetRepository,
)


class TestDetectionServiceSpecific:
    """Specific tests for DetectionService methods to improve coverage."""
    
    @pytest.fixture
    def detection_service(self):
        """Create DetectionService with repositories."""
        detector_repo = InMemoryDetectorRepository()
        result_repo = InMemoryResultRepository()
        scorer = AnomalyScorer()
        threshold_calc = ThresholdCalculator()
        
        return DetectionService(
            detector_repository=detector_repo,
            result_repository=result_repo,
            anomaly_scorer=scorer,
            threshold_calculator=threshold_calc
        )
    
    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_percentile(self, detection_service):
        """Test custom threshold detection with percentile method."""
        # Create detector
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1)
        )
        await detection_service.detector_repository.save(detector)
        
        # Create dataset
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        dataset = Dataset(name="test_dataset", features=features)
        
        # Mock the score method
        mock_scores = [AnomalyScore(v) for v in np.random.RandomState(42).beta(2, 8, 100)]
        with patch.object(detector, 'score', return_value=mock_scores):
            result = await detection_service.detect_with_custom_threshold(
                detector_id=detector.id,
                dataset=dataset,
                threshold_method="percentile",
                threshold_params={"percentile": 90}
            )
            
            assert isinstance(result, DetectionResult)
            assert result.threshold > 0
            assert "threshold_method" in result.metadata
            assert result.metadata["threshold_method"] == "percentile"
    
    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_iqr(self, detection_service):
        """Test custom threshold detection with IQR method."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await detection_service.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        mock_scores = [AnomalyScore(v) for v in np.random.random(50)]
        with patch.object(detector, 'score', return_value=mock_scores):
            result = await detection_service.detect_with_custom_threshold(
                detector_id=detector.id,
                dataset=dataset,
                threshold_method="iqr",
                threshold_params={"multiplier": 2.0}
            )
            
            assert result.metadata["threshold_method"] == "iqr"
            assert result.metadata["threshold_params"]["multiplier"] == 2.0
    
    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_mad(self, detection_service):
        """Test custom threshold detection with MAD method."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await detection_service.detector_repository.save(detector)
        
        features = np.random.random((30, 2))
        dataset = Dataset(name="test", features=features)
        
        mock_scores = [AnomalyScore(v) for v in np.random.random(30)]
        with patch.object(detector, 'score', return_value=mock_scores):
            result = await detection_service.detect_with_custom_threshold(
                detector_id=detector.id,
                dataset=dataset,
                threshold_method="mad",
                threshold_params={"factor": 3.5}
            )
            
            assert result.metadata["threshold_method"] == "mad"
            assert result.metadata["threshold_params"]["factor"] == 3.5
    
    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_dynamic(self, detection_service):
        """Test custom threshold detection with dynamic method."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await detection_service.detector_repository.save(detector)
        
        features = np.random.random((75, 4))
        dataset = Dataset(name="test", features=features)
        
        mock_scores = [AnomalyScore(v) for v in np.random.random(75)]
        with patch.object(detector, 'score', return_value=mock_scores):
            # Mock the dynamic threshold calculation
            with patch.object(detection_service.threshold_calculator, 'calculate_dynamic_threshold') as mock_dynamic:
                mock_dynamic.return_value = (0.7, {"knee_point": 0.7})
                
                result = await detection_service.detect_with_custom_threshold(
                    detector_id=detector.id,
                    dataset=dataset,
                    threshold_method="dynamic",
                    threshold_params={"method": "knee"}
                )
                
                assert result.metadata["threshold_method"] == "dynamic"
                mock_dynamic.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recompute_with_confidence_bootstrap(self, detection_service):
        """Test recomputing results with bootstrap confidence intervals."""
        # Create and save a detection result
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        scores = [AnomalyScore(v) for v in np.random.random(50)]
        result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=[],
            scores=scores,
            labels=np.array([0] * 50),
            threshold=0.5
        )
        
        await detection_service.result_repository.save(result)
        
        # Mock the confidence interval calculation
        with patch.object(detection_service.anomaly_scorer, 'add_confidence_intervals') as mock_ci:
            mock_scores_with_ci = []
            for score in scores:
                score_with_ci = AnomalyScore(score.value)
                score_with_ci.confidence_lower = score.value - 0.1
                score_with_ci.confidence_upper = score.value + 0.1
                score_with_ci.is_confident = True
                mock_scores_with_ci.append(score_with_ci)
            
            mock_ci.return_value = mock_scores_with_ci
            
            updated_result = await detection_service.recompute_with_confidence(
                result_id=result.id,
                confidence_level=0.95,
                method="bootstrap"
            )
            
            assert updated_result.confidence_intervals is not None
            assert len(updated_result.confidence_intervals) > 0
            assert "confidence_level" in updated_result.metadata
            assert updated_result.metadata["confidence_level"] == 0.95
    
    @pytest.mark.asyncio
    async def test_get_detection_history_by_detector(self, detection_service):
        """Test getting detection history filtered by detector."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create multiple results
        results = []
        for i in range(5):
            result = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=[AnomalyScore(0.5)],
                labels=np.array([0]),
                threshold=0.5,
                timestamp=datetime.now() - timedelta(hours=i)
            )
            await detection_service.result_repository.save(result)
            results.append(result)
        
        # Get history by detector
        history = await detection_service.get_detection_history(
            detector_id=detector_id,
            limit=3
        )
        
        assert len(history) == 3
        assert all(r.detector_id == detector_id for r in history)
        # Should be sorted by timestamp (newest first)
        for i in range(len(history) - 1):
            assert history[i].timestamp >= history[i + 1].timestamp
    
    @pytest.mark.asyncio
    async def test_get_detection_history_by_dataset(self, detection_service):
        """Test getting detection history filtered by dataset."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create results for specific dataset
        for i in range(3):
            result = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=[AnomalyScore(0.5)],
                labels=np.array([0]),
                threshold=0.5
            )
            await detection_service.result_repository.save(result)
        
        history = await detection_service.get_detection_history(
            dataset_id=dataset_id,
            limit=5
        )
        
        assert len(history) == 3
        assert all(r.dataset_id == dataset_id for r in history)
    
    @pytest.mark.asyncio
    async def test_compare_detectors_success(self, detection_service):
        """Test successful detector comparison."""
        # Create detectors
        detector_ids = []
        for i in range(3):
            detector = Detector(
                name=f"detector_{i}",
                algorithm=f"algo_{i}",
                contamination=ContaminationRate(0.1)
            )
            await detection_service.detector_repository.save(detector)
            detector_ids.append(detector.id)
        
        # Create dataset with labels
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
        dataset = Dataset(name="test_dataset", features=features, targets=targets)
        
        # Mock detection results
        with patch.object(detection_service, 'detect_with_multiple_detectors') as mock_detect:
            mock_results = {}
            for i, detector_id in enumerate(detector_ids):
                scores = [AnomalyScore(v) for v in np.random.RandomState(42 + i).random(100)]
                labels = np.random.RandomState(42 + i).choice([0, 1], size=100, p=[0.9, 0.1])
                
                result = DetectionResult(
                    detector_id=detector_id,
                    dataset_id=dataset.id,
                    anomalies=[],
                    scores=scores,
                    labels=labels,
                    threshold=0.5
                )
                mock_results[detector_id] = result
            
            mock_detect.return_value = mock_results
            
            comparison = await detection_service.compare_detectors(
                detector_ids=detector_ids,
                dataset=dataset,
                metrics=["precision", "recall", "f1", "auc_roc"]
            )
            
            assert "detectors" in comparison
            assert "summary" in comparison
            assert len(comparison["detectors"]) == 3
            
            # Check that all metrics are present
            for detector_result in comparison["detectors"].values():
                assert "precision" in detector_result
                assert "recall" in detector_result
                assert "f1" in detector_result
                assert "auc_roc" in detector_result
    
    @pytest.mark.asyncio
    async def test_compare_detectors_no_labels(self, detection_service):
        """Test detector comparison with dataset without labels."""
        detector_id = uuid4()
        features = np.random.random((50, 3))
        dataset = Dataset(name="unlabeled", features=features)  # No targets
        
        with pytest.raises(ValueError, match="Dataset must have labels for comparison"):
            await detection_service.compare_detectors([detector_id], dataset)


class TestModelPersistenceServiceSpecific:
    """Specific tests for ModelPersistenceService methods to improve coverage."""
    
    @pytest.fixture
    def persistence_service(self):
        """Create ModelPersistenceService with temporary storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            detector_repo = InMemoryDetectorRepository()
            service = ModelPersistenceService(
                detector_repository=detector_repo,
                storage_path=Path(tmp_dir)
            )
            yield service
    
    @pytest.mark.asyncio
    async def test_save_model_joblib_format(self, persistence_service):
        """Test saving model in joblib format."""
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1)
        )
        detector._is_fitted = True
        await persistence_service.detector_repository.save(detector)
        
        with patch('joblib.dump') as mock_dump:
            model_path = await persistence_service.save_model(
                detector_id=detector.id,
                format="joblib",
                metadata={"version": "1.0"}
            )
            
            assert model_path.endswith("model.joblib")
            mock_dump.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_model_onnx_format_not_implemented(self, persistence_service):
        """Test that ONNX format raises NotImplementedError."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await persistence_service.detector_repository.save(detector)
        
        with pytest.raises(NotImplementedError, match="ONNX format not yet supported"):
            await persistence_service.save_model(detector_id=detector.id, format="onnx")
    
    @pytest.mark.asyncio
    async def test_save_model_unknown_format(self, persistence_service):
        """Test saving model with unknown format."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await persistence_service.detector_repository.save(detector)
        
        with pytest.raises(ValueError, match="Unknown format: unknown"):
            await persistence_service.save_model(detector_id=detector.id, format="unknown")
    
    @pytest.mark.asyncio
    async def test_save_model_not_fitted(self, persistence_service):
        """Test saving model that is not fitted."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = False  # Not fitted
        await persistence_service.detector_repository.save(detector)
        
        with pytest.raises(ValueError, match="is not fitted"):
            await persistence_service.save_model(detector_id=detector.id)
    
    @pytest.mark.asyncio
    async def test_load_model_joblib_format(self, persistence_service):
        """Test loading model in joblib format."""
        detector_id = uuid4()
        
        # Create model directory
        model_dir = persistence_service.storage_path / str(detector_id)
        model_dir.mkdir()
        (model_dir / "model.joblib").touch()
        
        mock_detector = Mock()
        with patch('joblib.load', return_value=mock_detector) as mock_load:
            loaded_detector = await persistence_service.load_model(
                detector_id=detector_id,
                format="joblib"
            )
            
            assert loaded_detector == mock_detector
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_unknown_format(self, persistence_service):
        """Test loading model with unknown format."""
        detector_id = uuid4()
        model_dir = persistence_service.storage_path / str(detector_id)
        model_dir.mkdir()
        
        with pytest.raises(ValueError, match="Unknown format: unknown"):
            await persistence_service.load_model(detector_id=detector_id, format="unknown")
    
    @pytest.mark.asyncio
    async def test_export_model_comprehensive(self, persistence_service):
        """Test comprehensive model export."""
        detector = Detector(
            name="Production Detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.08)
        )
        detector._is_fitted = True
        await persistence_service.detector_repository.save(detector)
        
        export_path = persistence_service.storage_path / "export"
        
        with patch('pickle.dump') as mock_dump:
            exported_files = await persistence_service.export_model(
                detector_id=detector.id,
                export_path=export_path,
                include_data=False
            )
            
            assert "model" in exported_files
            assert "config" in exported_files
            assert "requirements" in exported_files
            assert "deploy_script" in exported_files
            
            # Verify files were created
            assert export_path.exists()
            mock_dump.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_saved_models_with_metadata(self, persistence_service):
        """Test listing saved models with metadata."""
        # Create mock model directories with metadata
        model_ids = [str(uuid4()) for _ in range(3)]
        
        for i, model_id in enumerate(model_ids):
            model_dir = persistence_service.storage_path / model_id
            model_dir.mkdir()
            
            metadata = {
                "detector_id": model_id,
                "detector_name": f"detector_{i}",
                "algorithm": f"algorithm_{i}",
                "saved_at": datetime.now().isoformat()
            }
            
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
        
        saved_models = await persistence_service.list_saved_models()
        
        assert len(saved_models) == 3
        for model_id in model_ids:
            assert model_id in saved_models
            assert "detector_name" in saved_models[model_id]
            assert "algorithm" in saved_models[model_id]
    
    def test_generate_requirements_algorithm_specific(self, persistence_service):
        """Test generating algorithm-specific requirements."""
        # Test PyOD detector
        pyod_detector = Mock()
        pyod_detector.algorithm_name = "PyOD_IsolationForest"
        
        requirements = persistence_service._generate_requirements(pyod_detector)
        assert "pyod>=2.0.5" in requirements
        
        # Test TODS detector
        tods_detector = Mock()
        tods_detector.algorithm_name = "TODS_LOF"
        
        requirements = persistence_service._generate_requirements(tods_detector)
        assert "tods>=0.1.0" in requirements
        
        # Test PyGOD detector
        pygod_detector = Mock()
        pygod_detector.algorithm_name = "PyGOD_GCNAE"
        
        requirements = persistence_service._generate_requirements(pygod_detector)
        assert "pygod>=1.0.0" in requirements
    
    def test_generate_deployment_script_content(self, persistence_service):
        """Test that deployment script contains expected content."""
        detector = Mock()
        detector.name = "Test Detector"
        detector.algorithm_name = "IsolationForest"
        
        script_content = persistence_service._generate_deployment_script(detector)
        
        assert "TestDetectorDetector" in script_content
        assert "detect" in script_content
        assert "predict_proba" in script_content
        assert "import pickle" in script_content
        assert "pd.DataFrame" in script_content


class TestExperimentTrackingServiceSpecific:
    """Specific tests for ExperimentTrackingService methods to improve coverage."""
    
    @pytest.fixture
    def tracking_service(self):
        """Create ExperimentTrackingService with temporary storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ExperimentTrackingService(tracking_path=Path(tmp_dir))
            yield service
    
    @pytest.mark.asyncio
    async def test_create_experiment_with_tags(self, tracking_service):
        """Test creating experiment with tags and description."""
        experiment_id = await tracking_service.create_experiment(
            name="Tagged Experiment",
            description="This is a test experiment with tags",
            tags=["anomaly_detection", "testing", "production"]
        )
        
        assert experiment_id in tracking_service.experiments
        experiment = tracking_service.experiments[experiment_id]
        assert experiment["name"] == "Tagged Experiment"
        assert experiment["description"] == "This is a test experiment with tags"
        assert len(experiment["tags"]) == 3
        assert "anomaly_detection" in experiment["tags"]
    
    @pytest.mark.asyncio
    async def test_log_run_with_artifacts(self, tracking_service):
        """Test logging a run with artifacts."""
        experiment_id = await tracking_service.create_experiment("Test Experiment")
        
        run_id = await tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="fraud_data",
            parameters={"n_estimators": 100, "contamination": 0.1},
            metrics={"accuracy": 0.95, "f1": 0.87},
            artifacts={"model": "/path/to/model.pkl", "plots": "/path/to/plots.png"}
        )
        
        experiment = tracking_service.experiments[experiment_id]
        run = next(r for r in experiment["runs"] if r["id"] == run_id)
        
        assert run["detector_name"] == "IsolationForest"
        assert run["artifacts"]["model"] == "/path/to/model.pkl"
        assert run["metrics"]["accuracy"] == 0.95
    
    @pytest.mark.asyncio
    async def test_compare_runs_specific_runs(self, tracking_service):
        """Test comparing specific runs within an experiment."""
        experiment_id = await tracking_service.create_experiment("Comparison Test")
        
        # Log multiple runs
        run_ids = []
        for i in range(5):
            run_id = await tracking_service.log_run(
                experiment_id=experiment_id,
                detector_name=f"detector_{i}",
                dataset_name="test_data",
                parameters={"param": i},
                metrics={"f1": 0.8 + i * 0.02}
            )
            run_ids.append(run_id)
        
        # Compare only first 3 runs
        comparison_df = await tracking_service.compare_runs(
            experiment_id=experiment_id,
            run_ids=run_ids[:3],
            metric="f1"
        )
        
        assert len(comparison_df) == 3
        assert "f1" in comparison_df.columns
        assert "param_param" in comparison_df.columns
        # Should be sorted by f1 score (descending)
        assert comparison_df.iloc[0]["f1"] >= comparison_df.iloc[1]["f1"]
    
    @pytest.mark.asyncio
    async def test_get_best_run_lower_is_better(self, tracking_service):
        """Test getting best run when lower values are better."""
        experiment_id = await tracking_service.create_experiment("Best Run Test")
        
        # Log runs with different error rates
        run_ids = []
        error_rates = [0.15, 0.08, 0.12, 0.06, 0.20]
        for i, error_rate in enumerate(error_rates):
            run_id = await tracking_service.log_run(
                experiment_id=experiment_id,
                detector_name=f"detector_{i}",
                dataset_name="test",
                parameters={},
                metrics={"error_rate": error_rate}
            )
            run_ids.append(run_id)
        
        best_run = await tracking_service.get_best_run(
            experiment_id=experiment_id,
            metric="error_rate",
            higher_is_better=False
        )
        
        assert best_run["metrics"]["error_rate"] == 0.06  # Lowest error rate
    
    @pytest.mark.asyncio
    async def test_get_best_run_no_metric(self, tracking_service):
        """Test getting best run when metric doesn't exist."""
        experiment_id = await tracking_service.create_experiment("No Metric Test")
        
        await tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="test_detector",
            dataset_name="test",
            parameters={},
            metrics={"accuracy": 0.95}  # Different metric
        )
        
        with pytest.raises(ValueError, match="No runs found with metric nonexistent"):
            await tracking_service.get_best_run(
                experiment_id=experiment_id,
                metric="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_log_artifact_for_run(self, tracking_service):
        """Test logging artifacts for specific runs."""
        experiment_id = await tracking_service.create_experiment("Artifact Test")
        
        run_id = await tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="test_detector",
            dataset_name="test",
            parameters={},
            metrics={"f1": 0.85}
        )
        
        await tracking_service.log_artifact(
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_name="confusion_matrix",
            artifact_path="/path/to/confusion_matrix.png"
        )
        
        # Find the run and check artifact
        experiment = tracking_service.experiments[experiment_id]
        run = next(r for r in experiment["runs"] if r["id"] == run_id)
        
        assert "confusion_matrix" in run["artifacts"]
        assert run["artifacts"]["confusion_matrix"] == "/path/to/confusion_matrix.png"
    
    @pytest.mark.asyncio
    async def test_create_leaderboard_all_experiments(self, tracking_service):
        """Test creating leaderboard across all experiments."""
        # Create multiple experiments
        exp_ids = []
        for i in range(3):
            exp_id = await tracking_service.create_experiment(f"Experiment_{i}")
            exp_ids.append(exp_id)
            
            # Log runs with different f1 scores
            await tracking_service.log_run(
                experiment_id=exp_id,
                detector_name=f"detector_{i}",
                dataset_name="test",
                parameters={},
                metrics={"f1": 0.8 + i * 0.05}
            )
        
        # Create leaderboard for all experiments (None = all)
        leaderboard = await tracking_service.create_leaderboard(
            experiment_ids=None,
            metric="f1"
        )
        
        assert len(leaderboard) == 3
        assert "rank" in leaderboard.columns
        assert leaderboard.iloc[0]["f1"] >= leaderboard.iloc[1]["f1"]  # Sorted descending
    
    @pytest.mark.asyncio
    async def test_export_experiment_complete(self, tracking_service):
        """Test complete experiment export."""
        experiment_id = await tracking_service.create_experiment(
            "Export Test",
            description="Test experiment for export"
        )
        
        # Log some runs
        for i in range(2):
            await tracking_service.log_run(
                experiment_id=experiment_id,
                detector_name=f"detector_{i}",
                dataset_name="test",
                parameters={"param": i},
                metrics={"f1": 0.8 + i * 0.1}
            )
        
        export_path = tracking_service.tracking_path / "export"
        
        await tracking_service.export_experiment(
            experiment_id=experiment_id,
            export_path=export_path
        )
        
        # Check that files were created
        assert (export_path / "experiment.json").exists()
        assert (export_path / "comparison.csv").exists()
        assert (export_path / "report.md").exists()
        
        # Check report content
        report_content = (export_path / "report.md").read_text()
        assert "Export Test" in report_content
        assert "Total runs: 2" in report_content
    
    def test_generate_experiment_report_content(self, tracking_service):
        """Test that experiment report contains expected content."""
        experiment = {
            "id": "test_id",
            "name": "Test Experiment",
            "created_at": "2024-01-01T00:00:00",
            "description": "Test description",
            "tags": ["tag1", "tag2"],
            "runs": [
                {
                    "id": "run1",
                    "detector_name": "IsolationForest",
                    "dataset_name": "test_data",
                    "timestamp": "2024-01-01T01:00:00",
                    "metrics": {"f1": 0.85, "precision": 0.80},
                    "parameters": {"n_estimators": 100}
                },
                {
                    "id": "run2",
                    "detector_name": "LOF",
                    "dataset_name": "test_data",
                    "timestamp": "2024-01-01T02:00:00",
                    "metrics": {"f1": 0.90, "precision": 0.88},
                    "parameters": {"n_neighbors": 20}
                }
            ]
        }
        
        report = tracking_service._generate_experiment_report(experiment)
        
        assert "# Experiment: Test Experiment" in report
        assert "**ID**: test_id" in report
        assert "Total runs: 2" in report
        assert "| f1 | 0.9000 |" in report  # Best f1 score
        assert "Run 1: IsolationForest" in report
        assert "Run 2: LOF" in report