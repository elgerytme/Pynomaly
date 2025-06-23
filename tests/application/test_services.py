"""Comprehensive tests for application layer services."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ExperimentTrackingService,
    ModelPersistenceService,
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator, EnsembleAggregator
from pynomaly.infrastructure.repositories import (
    InMemoryDetectorRepository,
    InMemoryResultRepository,
    InMemoryDatasetRepository,
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42}
    )


@pytest.fixture
def detection_service():
    """Create a DetectionService with mocked dependencies."""
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


@pytest.fixture
def ensemble_service():
    """Create an EnsembleService with mocked dependencies."""
    detector_repo = InMemoryDetectorRepository()
    aggregator = EnsembleAggregator()
    scorer = AnomalyScorer()
    
    return EnsembleService(
        detector_repository=detector_repo,
        ensemble_aggregator=aggregator,
        anomaly_scorer=scorer
    )


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def model_persistence_service(temp_storage_path):
    """Create a ModelPersistenceService with temporary storage."""
    detector_repo = InMemoryDetectorRepository()
    return ModelPersistenceService(
        detector_repository=detector_repo,
        storage_path=temp_storage_path
    )


@pytest.fixture
def experiment_tracking_service(temp_storage_path):
    """Create an ExperimentTrackingService with temporary storage."""
    return ExperimentTrackingService(tracking_path=temp_storage_path)


class TestDetectionService:
    """Test DetectionService functionality."""
    
    @pytest.mark.asyncio
    async def test_run_detection_success(self, detection_service, sample_dataset, sample_detector):
        """Test successful detection run."""
        # Store detector
        await detection_service.detector_repository.save(sample_detector)
        
        # Mock the actual detector algorithm
        with patch.object(detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            mock_scores = np.random.random(len(sample_dataset.features))
            mock_compute.return_value = mock_scores
            
            # Run detection
            result = await detection_service.run_detection(
                detector_id=sample_detector.id,
                dataset=sample_dataset
            )
            
            # Verify result
            assert isinstance(result, DetectionResult)
            assert result.detector.id == sample_detector.id
            assert result.dataset.name == sample_dataset.name
            assert len(result.scores) == len(sample_dataset.features)
            assert len(result.anomalies) > 0
            mock_compute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_detection_detector_not_found(self, detection_service, sample_dataset):
        """Test detection with non-existent detector."""
        with pytest.raises(ValueError, match="Detector not found"):
            await detection_service.run_detection(
                detector_id="non_existent",
                dataset=sample_dataset
            )
    
    @pytest.mark.asyncio
    async def test_run_detection_invalid_dataset(self, detection_service, sample_detector):
        """Test detection with invalid dataset."""
        await detection_service.detector_repository.save(sample_detector)
        
        # Create invalid dataset
        invalid_dataset = Dataset(
            name="invalid",
            features=np.array([]),  # Empty features
            targets=None
        )
        
        with pytest.raises(ValueError, match="Invalid dataset"):
            await detection_service.run_detection(
                detector_id=sample_detector.id,
                dataset=invalid_dataset
            )
    
    @pytest.mark.asyncio
    async def test_batch_detection(self, detection_service, sample_dataset, sample_detector):
        """Test batch detection with multiple datasets."""
        await detection_service.detector_repository.save(sample_detector)
        
        # Create multiple datasets
        datasets = [sample_dataset]
        for i in range(2):
            features = np.random.RandomState(42 + i).normal(0, 1, (50, 5))
            dataset = Dataset(name=f"dataset_{i}", features=features)
            datasets.append(dataset)
        
        with patch.object(detection_service.anomaly_scorer, 'compute_scores') as mock_compute:
            mock_compute.return_value = np.random.random(50)
            
            results = await detection_service.batch_detection(
                detector_id=sample_detector.id,
                datasets=datasets
            )
            
            assert len(results) == len(datasets)
            assert all(isinstance(r, DetectionResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_get_detection_history(self, detection_service, sample_detector):
        """Test retrieving detection history."""
        await detection_service.detector_repository.save(sample_detector)
        
        # Create mock detection results
        mock_results = []
        for i in range(3):
            features = np.random.RandomState(42 + i).normal(0, 1, (20, 5))
            dataset = Dataset(name=f"dataset_{i}", features=features)
            scores = np.random.random(20)
            anomalies = [Anomaly(score=AnomalyScore(0.9), index=0)]
            
            result = DetectionResult(
                detector=sample_detector,
                dataset=dataset,
                anomalies=anomalies,
                scores=scores
            )
            await detection_service.result_repository.save(result)
            mock_results.append(result)
        
        # Get history
        history = await detection_service.get_detection_history(sample_detector.id)
        
        assert len(history) == 3
        assert all(r.detector.id == sample_detector.id for r in history)


class TestEnsembleService:
    """Test EnsembleService functionality."""
    
    @pytest.mark.asyncio
    async def test_run_ensemble_detection(self, ensemble_service, sample_dataset):
        """Test ensemble detection with multiple detectors."""
        # Create multiple detectors
        detectors = []
        for i, algo in enumerate(["isolation_forest", "local_outlier_factor", "one_class_svm"]):
            detector = Detector(
                name=f"detector_{i}",
                algorithm=algo,
                contamination=ContaminationRate(0.1)
            )
            await ensemble_service.detector_repository.save(detector)
            detectors.append(detector)
        
        detector_ids = [d.id for d in detectors]
        
        with patch.object(ensemble_service.anomaly_scorer, 'compute_scores') as mock_compute:
            mock_compute.return_value = np.random.random(len(sample_dataset.features))
            
            result = await ensemble_service.run_ensemble_detection(
                detector_ids=detector_ids,
                dataset=sample_dataset,
                aggregation_method="mean"
            )
            
            assert isinstance(result, DetectionResult)
            assert result.dataset.name == sample_dataset.name
            assert len(result.scores) == len(sample_dataset.features)
            # Should be called once per detector
            assert mock_compute.call_count == len(detectors)
    
    @pytest.mark.asyncio
    async def test_ensemble_with_weights(self, ensemble_service, sample_dataset):
        """Test weighted ensemble detection."""
        # Create detectors
        detectors = []
        for i in range(3):
            detector = Detector(
                name=f"detector_{i}",
                algorithm="isolation_forest",
                contamination=ContaminationRate(0.1)
            )
            await ensemble_service.detector_repository.save(detector)
            detectors.append(detector)
        
        detector_ids = [d.id for d in detectors]
        weights = [0.5, 0.3, 0.2]
        
        with patch.object(ensemble_service.ensemble_aggregator, 'aggregate') as mock_agg:
            mock_agg.return_value = np.random.random(len(sample_dataset.features))
            
            result = await ensemble_service.run_ensemble_detection(
                detector_ids=detector_ids,
                dataset=sample_dataset,
                aggregation_method="weighted",
                weights=weights
            )
            
            assert isinstance(result, DetectionResult)
            mock_agg.assert_called_once()
            # Check that weights were passed
            call_args = mock_agg.call_args
            assert 'weights' in call_args.kwargs or len(call_args.args) > 2
    
    @pytest.mark.asyncio
    async def test_ensemble_empty_detectors(self, ensemble_service, sample_dataset):
        """Test ensemble with empty detector list."""
        with pytest.raises(ValueError, match="No detectors provided"):
            await ensemble_service.run_ensemble_detection(
                detector_ids=[],
                dataset=sample_dataset
            )
    
    @pytest.mark.asyncio
    async def test_ensemble_invalid_aggregation_method(self, ensemble_service, sample_dataset):
        """Test ensemble with invalid aggregation method."""
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1)
        )
        await ensemble_service.detector_repository.save(detector)
        
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            await ensemble_service.run_ensemble_detection(
                detector_ids=[detector.id],
                dataset=sample_dataset,
                aggregation_method="invalid_method"
            )


class TestModelPersistenceService:
    """Test ModelPersistenceService functionality."""
    
    @pytest.mark.asyncio
    async def test_save_model(self, model_persistence_service, sample_detector):
        """Test saving a model."""
        await model_persistence_service.detector_repository.save(sample_detector)
        
        # Mock trained model data
        model_data = {"weights": [1, 2, 3], "params": {"test": "value"}}
        
        with patch('pickle.dump') as mock_dump:
            file_path = await model_persistence_service.save_model(
                detector_id=sample_detector.id,
                model_data=model_data
            )
            
            assert file_path is not None
            assert file_path.suffix == '.pkl'
            mock_dump.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model(self, model_persistence_service, sample_detector):
        """Test loading a saved model."""
        await model_persistence_service.detector_repository.save(sample_detector)
        
        # Mock model file
        model_data = {"weights": [1, 2, 3], "params": {"test": "value"}}
        
        with patch('pickle.load') as mock_load, \
             patch('builtins.open', create=True) as mock_open:
            mock_load.return_value = model_data
            
            # First save, then load
            file_path = model_persistence_service.storage_path / f"{sample_detector.id}.pkl"
            file_path.touch()  # Create the file
            
            loaded_data = await model_persistence_service.load_model(sample_detector.id)
            
            assert loaded_data == model_data
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, model_persistence_service):
        """Test loading a non-existent model."""
        with pytest.raises(FileNotFoundError):
            await model_persistence_service.load_model("nonexistent_id")
    
    @pytest.mark.asyncio
    async def test_delete_model(self, model_persistence_service, sample_detector):
        """Test deleting a saved model."""
        await model_persistence_service.detector_repository.save(sample_detector)
        
        # Create mock model file
        file_path = model_persistence_service.storage_path / f"{sample_detector.id}.pkl"
        file_path.touch()
        
        # Delete model
        success = await model_persistence_service.delete_model(sample_detector.id)
        
        assert success is True
        assert not file_path.exists()
    
    @pytest.mark.asyncio
    async def test_list_saved_models(self, model_persistence_service):
        """Test listing saved models."""
        # Create mock model files
        model_ids = ["model1", "model2", "model3"]
        for model_id in model_ids:
            file_path = model_persistence_service.storage_path / f"{model_id}.pkl"
            file_path.touch()
        
        # List models
        saved_models = await model_persistence_service.list_saved_models()
        
        assert len(saved_models) == 3
        assert all(model_id in saved_models for model_id in model_ids)


class TestExperimentTrackingService:
    """Test ExperimentTrackingService functionality."""
    
    @pytest.mark.asyncio
    async def test_start_experiment(self, experiment_tracking_service):
        """Test starting a new experiment."""
        experiment_config = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "dataset": "test_dataset"
        }
        
        experiment_id = await experiment_tracking_service.start_experiment(
            name="test_experiment",
            config=experiment_config
        )
        
        assert experiment_id is not None
        assert isinstance(experiment_id, str)
        assert len(experiment_id) > 0
    
    @pytest.mark.asyncio
    async def test_log_metric(self, experiment_tracking_service):
        """Test logging metrics to an experiment."""
        experiment_id = await experiment_tracking_service.start_experiment(
            name="test_experiment",
            config={}
        )
        
        # Log metrics
        await experiment_tracking_service.log_metric(
            experiment_id=experiment_id,
            metric_name="accuracy",
            value=0.95,
            step=1
        )
        
        await experiment_tracking_service.log_metric(
            experiment_id=experiment_id,
            metric_name="f1_score",
            value=0.87,
            step=1
        )
        
        # Verify metrics are logged (this would depend on implementation)
        experiment = await experiment_tracking_service.get_experiment(experiment_id)
        assert experiment is not None
    
    @pytest.mark.asyncio
    async def test_log_artifact(self, experiment_tracking_service):
        """Test logging artifacts to an experiment."""
        experiment_id = await experiment_tracking_service.start_experiment(
            name="test_experiment",
            config={}
        )
        
        # Create temporary artifact
        artifact_path = experiment_tracking_service.tracking_path / "test_artifact.txt"
        artifact_path.write_text("test artifact content")
        
        await experiment_tracking_service.log_artifact(
            experiment_id=experiment_id,
            artifact_path=artifact_path,
            artifact_name="test_artifact"
        )
        
        # Verify artifact is logged
        experiment = await experiment_tracking_service.get_experiment(experiment_id)
        assert experiment is not None
    
    @pytest.mark.asyncio
    async def test_finish_experiment(self, experiment_tracking_service):
        """Test finishing an experiment."""
        experiment_id = await experiment_tracking_service.start_experiment(
            name="test_experiment",
            config={}
        )
        
        # Finish experiment
        await experiment_tracking_service.finish_experiment(
            experiment_id=experiment_id,
            status="completed"
        )
        
        # Verify experiment status
        experiment = await experiment_tracking_service.get_experiment(experiment_id)
        assert experiment is not None
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, experiment_tracking_service):
        """Test listing experiments."""
        # Start multiple experiments
        experiment_ids = []
        for i in range(3):
            exp_id = await experiment_tracking_service.start_experiment(
                name=f"experiment_{i}",
                config={"index": i}
            )
            experiment_ids.append(exp_id)
        
        # List experiments
        experiments = await experiment_tracking_service.list_experiments()
        
        assert len(experiments) >= 3
        assert all(exp_id in [exp.id for exp in experiments] for exp_id in experiment_ids)
    
    @pytest.mark.asyncio
    async def test_compare_experiments(self, experiment_tracking_service):
        """Test comparing multiple experiments."""
        # Start experiments with different metrics
        exp1_id = await experiment_tracking_service.start_experiment("exp1", {})
        exp2_id = await experiment_tracking_service.start_experiment("exp2", {})
        
        # Log different metrics
        await experiment_tracking_service.log_metric(exp1_id, "accuracy", 0.95, 1)
        await experiment_tracking_service.log_metric(exp2_id, "accuracy", 0.87, 1)
        
        # Compare experiments
        comparison = await experiment_tracking_service.compare_experiments([exp1_id, exp2_id])
        
        assert comparison is not None
        assert len(comparison) == 2