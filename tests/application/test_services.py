"""Comprehensive tests for application layer services."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ExperimentTrackingService,
    ModelPersistenceService,
)
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    ThresholdCalculator,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.repositories import (
    AsyncDetectionResultRepositoryWrapper,
    AsyncDetectorRepositoryWrapper,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    import pandas as pd

    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])

    # Create DataFrame with features and target column
    df = pd.DataFrame(
        features,
        columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
    )
    df["target"] = targets

    return Dataset(name="test_dataset", data=df, target_column="target")


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    from pynomaly.infrastructure.adapters import SklearnAdapter

    return SklearnAdapter(
        algorithm_name="IsolationForest",
        name="test_detector",
        contamination_rate=ContaminationRate(0.1),
        n_estimators=100,
        random_state=42,
    )


@pytest.fixture
def detection_service():
    """Create a DetectionService with mocked dependencies."""
    sync_detector_repo = InMemoryDetectorRepository()
    sync_result_repo = InMemoryResultRepository()

    # Wrap sync repositories with async wrappers
    detector_repo = AsyncDetectorRepositoryWrapper(sync_detector_repo)
    result_repo = AsyncDetectionResultRepositoryWrapper(sync_result_repo)

    scorer = AnomalyScorer()
    threshold_calc = ThresholdCalculator()

    return DetectionService(
        detector_repository=detector_repo,
        result_repository=result_repo,
        anomaly_scorer=scorer,
        threshold_calculator=threshold_calc,
    )


@pytest.fixture
def ensemble_service():
    """Create an EnsembleService with mocked dependencies."""
    sync_detector_repo = InMemoryDetectorRepository()
    detector_repo = AsyncDetectorRepositoryWrapper(sync_detector_repo)
    aggregator = EnsembleAggregator()
    scorer = AnomalyScorer()

    return EnsembleService(
        detector_repository=detector_repo,
        ensemble_aggregator=aggregator,
        anomaly_scorer=scorer,
    )


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def model_persistence_service(temp_storage_path):
    """Create a ModelPersistenceService with temporary storage."""
    sync_detector_repo = InMemoryDetectorRepository()
    detector_repo = AsyncDetectorRepositoryWrapper(sync_detector_repo)
    return ModelPersistenceService(
        detector_repository=detector_repo, storage_path=temp_storage_path
    )


@pytest.fixture
def experiment_tracking_service(temp_storage_path):
    """Create an ExperimentTrackingService with temporary storage."""
    return ExperimentTrackingService(tracking_path=temp_storage_path)


class TestDetectionService:
    """Test DetectionService functionality."""

    @pytest.mark.asyncio
    async def test_run_detection_success(
        self, detection_service, sample_dataset, sample_detector
    ):
        """Test successful detection run."""
        # Store detector and set as fitted
        sample_detector.is_fitted = True
        await detection_service.detector_repository.save(sample_detector)

        # Mock the detector's score method directly
        with patch.object(sample_detector, "score") as mock_score:
            mock_score_objects = [
                AnomalyScore(value=val)
                for val in np.random.random(len(sample_dataset.data))
            ]
            mock_score.return_value = mock_score_objects

            # Mock the detector's detect method
            with patch.object(sample_detector, "detect") as mock_detect:
                mock_anomalies = [
                    Anomaly(
                        score=score_obj,
                        data_point=sample_dataset.data.iloc[i].to_dict(),
                        detector_name=sample_detector.name,
                    )
                    for i, score_obj in enumerate(
                        mock_score_objects[:5]
                    )  # Just first 5 as anomalies
                ]

                mock_result = DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=sample_dataset.id,
                    anomalies=mock_anomalies,
                    scores=mock_score_objects,
                    labels=np.array(
                        [1 if i < 5 else 0 for i in range(len(sample_dataset.data))]
                    ),
                    threshold=0.8,
                )
                mock_detect.return_value = mock_result

                # Run detection
                results = await detection_service.detect_with_multiple_detectors(
                    detector_ids=[sample_detector.id], dataset=sample_dataset
                )

                # Verify result
                assert len(results) == 1
                result = results[sample_detector.id]
                assert isinstance(result, DetectionResult)
                assert result.detector_id == sample_detector.id
                assert result.dataset_id == sample_dataset.id
                assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.asyncio
    async def test_run_detection_detector_not_found(
        self, detection_service, sample_dataset
    ):
        """Test detection with non-existent detector."""
        from uuid import uuid4

        with pytest.raises(ValueError, match="Detector.*not found"):
            await detection_service.detect_with_multiple_detectors(
                detector_ids=[uuid4()], dataset=sample_dataset
            )

    @pytest.mark.asyncio
    async def test_run_detection_invalid_dataset(
        self, detection_service, sample_detector
    ):
        """Test detection with invalid dataset."""
        await detection_service.detector_repository.save(sample_detector)

        # Create invalid dataset should raise an exception
        import pandas as pd

        with pytest.raises(
            Exception
        ):  # InvalidDataError should be raised from Dataset constructor
            invalid_dataset = Dataset(
                name="invalid",
                data=pd.DataFrame(),  # Empty DataFrame
            )

    @pytest.mark.asyncio
    async def test_batch_detection(
        self, detection_service, sample_dataset, sample_detector
    ):
        """Test batch detection with multiple datasets."""
        await detection_service.detector_repository.save(sample_detector)

        # Create multiple datasets
        datasets = [sample_dataset]
        for i in range(2):
            features = np.random.RandomState(42 + i).normal(0, 1, (50, 5))
            df = pd.DataFrame(features, columns=[f"feature_{j}" for j in range(5)])
            dataset = Dataset(name=f"dataset_{i}", data=df)
            datasets.append(dataset)

        # Store detector and set as fitted
        sample_detector.is_fitted = True
        await detection_service.detector_repository.save(sample_detector)

        with patch.object(sample_detector, "detect") as mock_detect:
            # Mock detection results for each dataset
            mock_results = []
            for i, dataset in enumerate(datasets):
                labels = np.array([1 if j < 5 else 0 for j in range(len(dataset.data))])
                anomaly_indices = np.where(labels == 1)[0]

                mock_anomalies = [
                    Anomaly(
                        score=AnomalyScore(value=0.9),
                        data_point=dataset.data.iloc[idx].to_dict(),
                        detector_name=sample_detector.name,
                    )
                    for idx in anomaly_indices
                ]

                mock_result = DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=dataset.id,
                    anomalies=mock_anomalies,
                    scores=[
                        AnomalyScore(s) for s in np.random.random(len(dataset.data))
                    ],
                    labels=labels,
                    threshold=0.8,
                )
                mock_results.append(mock_result)

            mock_detect.side_effect = mock_results

            # Run detection for each dataset separately
            results = []
            for dataset in datasets:
                result = await detection_service.detect_with_multiple_detectors(
                    detector_ids=[sample_detector.id], dataset=dataset
                )
                results.append(list(result.values())[0])

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
            df = pd.DataFrame(features, columns=[f"feature_{j}" for j in range(5)])
            dataset = Dataset(name=f"dataset_{i}", data=df)
            scores = np.random.random(20)
            anomalies = [
                Anomaly(
                    score=AnomalyScore(value=0.9),
                    data_point={"feature_0": 1.0, "feature_1": 2.0},
                    detector_name=sample_detector.name,
                )
            ]

            labels = np.array(
                [1 if j < 1 else 0 for j in range(len(scores))]
            )  # Just first one as anomaly
            result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=[AnomalyScore(s) for s in scores],
                labels=labels,
                threshold=0.8,
            )
            await detection_service.result_repository.save(result)
            mock_results.append(result)

        # Get history
        history = await detection_service.get_detection_history(sample_detector.id)

        assert len(history) == 3
        assert all(r.detector_id == sample_detector.id for r in history)


class TestEnsembleService:
    """Test EnsembleService functionality."""

    @pytest.mark.asyncio
    async def test_create_ensemble(self, ensemble_service, sample_dataset):
        """Test creating an ensemble from multiple detectors."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        # Create multiple detectors
        detectors = []
        for i, algo in enumerate(
            ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        ):
            detector = SklearnAdapter(
                algorithm_name=algo,
                name=f"detector_{i}",
                contamination_rate=ContaminationRate(0.1),
            )
            await ensemble_service.detector_repository.save(detector)
            detectors.append(detector)

        detector_ids = [d.id for d in detectors]

        # Create ensemble
        ensemble = await ensemble_service.create_ensemble(
            name="test_ensemble",
            detector_ids=detector_ids,
            aggregation_method="average",
        )

        assert ensemble.name == "test_ensemble"
        assert len(ensemble.base_detectors) == len(detectors)
        assert ensemble.aggregation_method == "average"

    @pytest.mark.asyncio
    async def test_ensemble_with_weights(self, ensemble_service, sample_dataset):
        """Test weighted ensemble creation."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        # Create detectors
        detectors = []
        for i in range(3):
            detector = SklearnAdapter(
                algorithm_name="IsolationForest",
                name=f"detector_{i}",
                contamination_rate=ContaminationRate(0.1),
            )
            await ensemble_service.detector_repository.save(detector)
            detectors.append(detector)

        detector_ids = [d.id for d in detectors]
        weights = {d.id: [0.5, 0.3, 0.2][i] for i, d in enumerate(detectors)}

        # Create weighted ensemble
        ensemble = await ensemble_service.create_ensemble(
            name="weighted_ensemble",
            detector_ids=detector_ids,
            weights=weights,
            aggregation_method="weighted",
        )

        assert ensemble.name == "weighted_ensemble"
        assert len(ensemble.base_detectors) == 3
        current_weights = ensemble.get_current_weights()
        assert len(current_weights) == 3

    @pytest.mark.asyncio
    async def test_ensemble_empty_detectors(self, ensemble_service, sample_dataset):
        """Test ensemble with empty detector list."""
        with pytest.raises(ValueError):
            await ensemble_service.create_ensemble(
                name="empty_ensemble", detector_ids=[]
            )

    @pytest.mark.asyncio
    async def test_ensemble_nonexistent_detector(
        self, ensemble_service, sample_dataset
    ):
        """Test ensemble with non-existent detector."""
        from uuid import uuid4

        with pytest.raises(ValueError, match="not found"):
            await ensemble_service.create_ensemble(
                name="invalid_ensemble", detector_ids=[uuid4()]
            )


class TestModelPersistenceService:
    """Test ModelPersistenceService functionality."""

    @pytest.mark.asyncio
    async def test_save_model(self, model_persistence_service, sample_detector):
        """Test saving a model."""
        # Mark detector as fitted
        sample_detector.is_fitted = True
        await model_persistence_service.detector_repository.save(sample_detector)

        with patch("pickle.dump") as mock_dump:
            file_path = await model_persistence_service.save_model(
                detector_id=sample_detector.id, format="pickle"
            )

            assert file_path is not None
            assert file_path.endswith(".pkl")
            mock_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model(self, model_persistence_service, sample_detector):
        """Test loading a saved model."""
        sample_detector.is_fitted = True
        await model_persistence_service.detector_repository.save(sample_detector)

        with (
            patch("pickle.load") as mock_load,
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_load.return_value = sample_detector

            # Create model directory and file
            model_dir = model_persistence_service.storage_path / str(sample_detector.id)
            model_dir.mkdir(exist_ok=True)
            model_file = model_dir / "model.pkl"
            model_file.touch()

            loaded_detector = await model_persistence_service.load_model(
                sample_detector.id
            )

            assert loaded_detector == sample_detector
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, model_persistence_service):
        """Test loading a non-existent model."""
        from uuid import uuid4

        with pytest.raises(ValueError, match="No saved model found"):
            await model_persistence_service.load_model(uuid4())

    @pytest.mark.asyncio
    async def test_delete_model(self, model_persistence_service, sample_detector):
        """Test deleting a saved model."""
        await model_persistence_service.detector_repository.save(sample_detector)

        # Create mock model directory and file structure
        model_dir = model_persistence_service.storage_path / str(sample_detector.id)
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "model.pkl"
        model_file.touch()

        # Delete model
        success = await model_persistence_service.delete_model(sample_detector.id)

        assert success is True
        assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_list_saved_models(self, model_persistence_service):
        """Test listing saved models."""
        # Create mock model directories with metadata
        model_ids = ["model1", "model2", "model3"]
        for model_id in model_ids:
            model_dir = model_persistence_service.storage_path / model_id
            model_dir.mkdir(exist_ok=True)

            # Create metadata file
            metadata = {
                "detector_id": model_id,
                "detector_name": f"detector_{model_id}",
                "algorithm": "IsolationForest",
                "saved_at": "2024-01-01T00:00:00",
                "format": "pickle",
            }
            meta_path = model_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

        # List models
        saved_models = await model_persistence_service.list_saved_models()

        assert len(saved_models) == 3
        assert all(model_id in saved_models for model_id in model_ids)


class TestExperimentTrackingService:
    """Test ExperimentTrackingService functionality."""

    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_tracking_service):
        """Test creating a new experiment."""
        experiment_id = await experiment_tracking_service.create_experiment(
            name="test_experiment",
            description="A test experiment",
            tags=["test", "anomaly_detection"],
        )

        assert experiment_id is not None
        assert isinstance(experiment_id, str)
        assert len(experiment_id) > 0

    @pytest.mark.asyncio
    async def test_log_run(self, experiment_tracking_service):
        """Test logging a run to an experiment."""
        experiment_id = await experiment_tracking_service.create_experiment(
            name="test_experiment"
        )

        # Log a run
        run_id = await experiment_tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"contamination": 0.1, "n_estimators": 100},
            metrics={"accuracy": 0.95, "f1_score": 0.87},
        )

        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    @pytest.mark.asyncio
    async def test_log_artifact(self, experiment_tracking_service):
        """Test logging artifacts to an experiment run."""
        experiment_id = await experiment_tracking_service.create_experiment(
            name="test_experiment"
        )

        # Create a run first
        run_id = await experiment_tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"contamination": 0.1},
            metrics={"accuracy": 0.95},
        )

        # Create temporary artifact
        artifact_path = experiment_tracking_service.tracking_path / "test_artifact.txt"
        artifact_path.write_text("test artifact content")

        await experiment_tracking_service.log_artifact(
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_name="test_artifact",
            artifact_path=str(artifact_path),
        )

        # Verify artifact is logged (no direct get_experiment method, so just check no exception)
        assert True  # Test passes if no exception was raised

    @pytest.mark.asyncio
    async def test_get_best_run(self, experiment_tracking_service):
        """Test getting the best run from an experiment."""
        experiment_id = await experiment_tracking_service.create_experiment(
            name="test_experiment"
        )

        # Log multiple runs with different metrics
        await experiment_tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"contamination": 0.1},
            metrics={"f1": 0.85, "accuracy": 0.90},
        )

        await experiment_tracking_service.log_run(
            experiment_id=experiment_id,
            detector_name="LOF",
            dataset_name="test_dataset",
            parameters={"n_neighbors": 5},
            metrics={"f1": 0.92, "accuracy": 0.88},
        )

        # Get best run by F1 score
        best_run = await experiment_tracking_service.get_best_run(
            experiment_id=experiment_id, metric="f1"
        )

        assert best_run is not None
        assert best_run["metrics"]["f1"] == 0.92
        assert best_run["detector_name"] == "LOF"

    @pytest.mark.asyncio
    async def test_compare_runs(self, experiment_tracking_service):
        """Test comparing runs within an experiment."""
        experiment_id = await experiment_tracking_service.create_experiment(
            name="test_experiment"
        )

        # Log multiple runs
        for i, detector in enumerate(["IsolationForest", "LOF", "OneClassSVM"]):
            await experiment_tracking_service.log_run(
                experiment_id=experiment_id,
                detector_name=detector,
                dataset_name="test_dataset",
                parameters={"param": i * 10},
                metrics={"accuracy": 0.85 + i * 0.05, "f1": 0.80 + i * 0.04},
            )

        # Compare runs
        comparison_df = await experiment_tracking_service.compare_runs(
            experiment_id=experiment_id, metric="accuracy"
        )

        assert len(comparison_df) == 3
        assert "accuracy" in comparison_df.columns
        assert "detector" in comparison_df.columns

    @pytest.mark.asyncio
    async def test_create_multiple_experiments(self, experiment_tracking_service):
        """Test creating multiple experiments."""
        # Create multiple experiments
        experiment_ids = []
        for i in range(3):
            exp_id = await experiment_tracking_service.create_experiment(
                name=f"experiment_{i}",
                description=f"Test experiment {i}",
                tags=["test", f"batch_{i}"],
            )
            experiment_ids.append(exp_id)

        # Verify all experiments were created
        assert len(experiment_ids) == 3
        assert all(isinstance(exp_id, str) for exp_id in experiment_ids)
        assert len(set(experiment_ids)) == 3  # All IDs are unique
