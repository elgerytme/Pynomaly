"""Tests for advanced ML lifecycle management service."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from pynomaly.application.services.advanced_ml_lifecycle_service import (
    AdvancedMLLifecycleService,
)
from pynomaly.domain.entities import (
    Experiment,
    ExperimentRun,
    ExperimentStatus,
    ExperimentType,
    Model,
    ModelStatus,
    ModelVersion,
)
from pynomaly.domain.value_objects import (
    ModelStorageInfo,
    PerformanceMetrics,
    SemanticVersion,
)


@pytest.fixture
def mock_repositories():
    """Create mock repositories."""
    experiment_repo = AsyncMock()
    model_repo = AsyncMock()
    model_version_repo = AsyncMock()
    return experiment_repo, model_repo, model_version_repo


@pytest.fixture
def temp_directories():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        artifact_path = temp_path / "artifacts"
        registry_path = temp_path / "registry"
        artifact_path.mkdir()
        registry_path.mkdir()
        yield artifact_path, registry_path


@pytest.fixture
def lifecycle_service(mock_repositories, temp_directories):
    """Create advanced ML lifecycle service."""
    experiment_repo, model_repo, model_version_repo = mock_repositories
    artifact_path, registry_path = temp_directories

    return AdvancedMLLifecycleService(
        experiment_repository=experiment_repo,
        model_repository=model_repo,
        model_version_repository=model_version_repo,
        artifact_storage_path=artifact_path,
        model_registry_path=registry_path,
    )


@pytest.fixture
def sample_experiment():
    """Create sample experiment."""
    return Experiment(
        name="Fraud Detection Experiment",
        description="Testing various algorithms for fraud detection",
        experiment_type=ExperimentType.ALGORITHM_COMPARISON,
        objective="Maximize F1 score while maintaining low false positive rate",
        created_by="data_scientist",
        tags=["fraud", "classification", "production"],
    )


@pytest.fixture
def sample_model():
    """Create sample model."""
    return Model(
        name="fraud_detector_v1",
        description="Production fraud detection model",
        model_type="anomaly_detection",
        algorithm_family="ensemble",
        created_by="ml_engineer",
        team="fraud_team",
    )


@pytest.fixture
def sample_model_version(sample_model):
    """Create sample model version."""
    performance_metrics = PerformanceMetrics(
        accuracy=0.92,
        precision=0.89,
        recall=0.94,
        f1_score=0.91,
        training_time=120.5,
        inference_time=15.2,
    )

    storage_info = ModelStorageInfo(
        storage_path="/tmp/model.pkl",
        storage_type="local_file",
        compression="none",
        size_bytes=1024,
        checksum="abc123",
    )

    return ModelVersion(
        model_id=sample_model.id,
        detector_id=uuid4(),
        version=SemanticVersion(major=1, minor=0, patch=0),
        performance_metrics=performance_metrics,
        storage_info=storage_info,
        created_by="ml_engineer",
        description="Initial production version",
    )


class TestExperimentTracking:
    """Test experiment tracking functionality."""

    @pytest.mark.asyncio
    async def test_start_experiment(self, lifecycle_service, mock_repositories):
        """Test starting a new experiment."""
        experiment_repo, _, _ = mock_repositories

        # Mock repository save
        experiment_repo.save = AsyncMock()

        experiment_id = await lifecycle_service.start_experiment(
            name="Test Experiment",
            description="Testing experiment tracking",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
            objective="Optimize model performance",
            created_by="test_user",
            tags=["test", "optimization"],
        )

        # Verify experiment was created
        assert experiment_id is not None
        assert experiment_id in lifecycle_service._active_experiments

        # Verify repository was called
        experiment_repo.save.assert_called_once()

        # Verify experiment properties
        active_exp = lifecycle_service._active_experiments[experiment_id]
        experiment = active_exp["experiment"]
        assert experiment.name == "Test Experiment"
        assert experiment.experiment_type == ExperimentType.HYPERPARAMETER_TUNING
        assert experiment.status == ExperimentStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_run(
        self, lifecycle_service, mock_repositories, sample_experiment
    ):
        """Test starting an experiment run."""
        experiment_repo, _, _ = mock_repositories

        # Setup experiment
        sample_experiment.start_experiment()
        experiment_repo.find_by_id.return_value = sample_experiment
        experiment_repo.save = AsyncMock()

        # Add experiment to active experiments
        exp_id = str(sample_experiment.id)
        lifecycle_service._active_experiments[exp_id] = {
            "experiment": sample_experiment,
            "start_time": datetime.utcnow(),
            "runs": [],
        }

        # Start run
        run_id = await lifecycle_service.start_run(
            experiment_id=exp_id,
            run_name="Test Run",
            detector_id=uuid4(),
            dataset_id=uuid4(),
            parameters={"contamination": 0.1, "n_estimators": 100},
            created_by="test_user",
            description="Testing run tracking",
        )

        # Verify run was created
        assert run_id is not None
        assert run_id in lifecycle_service._active_runs

        # Verify run properties
        run_info = lifecycle_service._active_runs[run_id]
        run = run_info["run"]
        assert run.name == "Test Run"
        assert run.status == "running"
        assert run.parameters["contamination"] == 0.1

    @pytest.mark.asyncio
    async def test_log_parameter(self, lifecycle_service):
        """Test logging parameters."""
        # Create mock run
        run_id = str(uuid4())
        lifecycle_service._active_runs[run_id] = {
            "run": MagicMock(),
            "logged_parameters": {},
        }
        lifecycle_service._active_runs[run_id]["run"].parameters = {}

        # Log parameter
        await lifecycle_service.log_parameter(run_id, "learning_rate", 0.01)

        # Verify parameter was logged
        assert (
            lifecycle_service._active_runs[run_id]["logged_parameters"]["learning_rate"]
            == 0.01
        )
        assert (
            lifecycle_service._active_runs[run_id]["run"].parameters["learning_rate"]
            == 0.01
        )

    @pytest.mark.asyncio
    async def test_log_metric(self, lifecycle_service):
        """Test logging metrics."""
        # Create mock run
        run_id = str(uuid4())
        lifecycle_service._active_runs[run_id] = {
            "run": MagicMock(),
            "logged_metrics": {},
        }
        lifecycle_service._active_runs[run_id]["run"].metrics = {}

        # Log metric
        await lifecycle_service.log_metric(run_id, "accuracy", 0.95, step=1)

        # Verify metric was logged
        metrics = lifecycle_service._active_runs[run_id]["logged_metrics"]
        assert "accuracy" in metrics
        assert len(metrics["accuracy"]) == 1
        assert metrics["accuracy"][0]["value"] == 0.95
        assert metrics["accuracy"][0]["step"] == 1

    @pytest.mark.asyncio
    async def test_log_artifact(self, lifecycle_service, temp_directories):
        """Test logging artifacts."""
        artifact_path, _ = temp_directories

        # Create mock run
        run_id = str(uuid4())
        run_dir = artifact_path / "exp1" / run_id
        run_dir.mkdir(parents=True)

        lifecycle_service._active_runs[run_id] = {
            "run": MagicMock(),
            "run_dir": run_dir,
            "logged_artifacts": {},
        }
        lifecycle_service._active_runs[run_id]["run"].artifacts = {}

        # Test different artifact types
        test_data = {"test": "data"}

        # JSON artifact
        artifact_path_json = await lifecycle_service.log_artifact(
            run_id, "test_data", test_data, "json"
        )
        assert Path(artifact_path_json).exists()

        # Verify artifact was logged
        artifacts = lifecycle_service._active_runs[run_id]["logged_artifacts"]
        assert "test_data" in artifacts
        assert artifacts["test_data"]["type"] == "json"

    @pytest.mark.asyncio
    async def test_log_model(self, lifecycle_service, temp_directories):
        """Test logging models."""
        artifact_path, _ = temp_directories

        # Create mock run
        run_id = str(uuid4())
        run_dir = artifact_path / "exp1" / run_id
        run_dir.mkdir(parents=True)

        lifecycle_service._active_runs[run_id] = {
            "run": MagicMock(),
            "run_dir": run_dir,
            "logged_artifacts": {},
        }
        lifecycle_service._active_runs[run_id]["run"].artifacts = {}

        # Create sample model
        model = IsolationForest(contamination=0.1)
        X = np.random.random((100, 5))
        model.fit(X)

        # Log model
        model_path = await lifecycle_service.log_model(
            run_id=run_id,
            model=model,
            model_name="isolation_forest",
            model_signature={
                "input_schema": "numpy_array",
                "output_schema": "predictions",
            },
        )

        # Verify model was logged
        assert Path(model_path).exists()
        artifacts = lifecycle_service._active_runs[run_id]["logged_artifacts"]
        assert "model_isolation_forest" in artifacts

    @pytest.mark.asyncio
    async def test_end_run(self, lifecycle_service, temp_directories):
        """Test ending an experiment run."""
        artifact_path, _ = temp_directories

        # Create mock run
        run_id = str(uuid4())
        run_dir = artifact_path / "exp1" / run_id
        run_dir.mkdir(parents=True)

        mock_run = MagicMock()
        mock_run.id = UUID(run_id)
        mock_run.name = "Test Run"
        mock_run.duration_seconds = 120.5
        mock_run.metrics = {"accuracy": 0.95}

        lifecycle_service._active_runs[run_id] = {
            "run": mock_run,
            "run_dir": run_dir,
            "logged_metrics": {"accuracy": [{"value": 0.95, "step": 1}]},
            "logged_parameters": {"contamination": 0.1},
            "logged_artifacts": {"model": {"path": "/tmp/model.pkl"}},
        }

        # End run
        completed_run = await lifecycle_service.end_run(run_id, "FINISHED")

        # Verify run was completed
        assert run_id not in lifecycle_service._active_runs
        assert (run_dir / "run_summary.json").exists()

        # Verify summary file
        with open(run_dir / "run_summary.json") as f:
            summary = json.load(f)
        assert summary["status"] == "FINISHED"
        assert summary["metrics"]["accuracy"] == 0.95


class TestModelVersioning:
    """Test model versioning functionality."""

    @pytest.mark.asyncio
    async def test_create_model_version_new_model(
        self, lifecycle_service, mock_repositories
    ):
        """Test creating a model version for a new model."""
        experiment_repo, model_repo, model_version_repo = mock_repositories

        # Mock repositories
        model_repo.find_by_name.return_value = []  # No existing model
        model_repo.save = AsyncMock()
        model_version_repo.save = AsyncMock()

        # Create mock run
        run_id = str(uuid4())
        mock_run = MagicMock()
        mock_run.metadata = {"created_by": "test_user"}
        mock_run.detector_id = uuid4()

        lifecycle_service._active_runs[run_id] = {
            "run": mock_run,
        }

        # Create model version
        version_id = await lifecycle_service.create_model_version(
            model_name="new_model",
            run_id=run_id,
            model_path="/tmp/model.pkl",
            performance_metrics={"accuracy": 0.95, "f1_score": 0.92},
            description="Initial version",
        )

        # Verify model and version were created
        assert version_id is not None
        model_repo.save.assert_called()
        model_version_repo.save.assert_called()

    @pytest.mark.asyncio
    async def test_determine_next_version(
        self, lifecycle_service, mock_repositories, sample_model_version
    ):
        """Test automatic version determination."""
        _, _, model_version_repo = mock_repositories

        # Mock existing versions
        model_version_repo.find_by_model_id.return_value = [sample_model_version]

        # Test significant improvement (minor version bump)
        new_metrics = {"accuracy": 0.98, "f1_score": 0.97}  # Significant improvement
        version = await lifecycle_service._determine_next_version(
            sample_model_version.model_id, new_metrics
        )
        assert version.minor == 1
        assert version.patch == 0

    @pytest.mark.asyncio
    async def test_promote_model_version(
        self, lifecycle_service, mock_repositories, sample_model_version
    ):
        """Test model version promotion."""
        _, _, model_version_repo = mock_repositories

        # Mock repository
        model_version_repo.find_by_id.return_value = sample_model_version
        model_version_repo.save = AsyncMock()

        # Promote to staging
        result = await lifecycle_service.promote_model_version(
            model_version_id=str(sample_model_version.id),
            stage="staging",
            promoted_by="ml_engineer",
            validation_tests=["performance_baseline"],
        )

        # Verify promotion
        assert result["success"] is True
        assert result["new_stage"] == "staging"
        assert result["new_status"] == "validated"
        model_version_repo.save.assert_called()


class TestModelRegistry:
    """Test model registry functionality."""

    @pytest.mark.asyncio
    async def test_search_models(
        self, lifecycle_service, mock_repositories, sample_model
    ):
        """Test model search functionality."""
        _, model_repo, model_version_repo = mock_repositories

        # Mock repository
        model_repo.find_all.return_value = [sample_model]
        model_version_repo.find_by_id.return_value = None

        # Search models
        results = await lifecycle_service.search_models(
            query="fraud",
            max_results=10,
        )

        # Verify results
        assert len(results) == 1
        assert results[0]["name"] == sample_model.name

    @pytest.mark.asyncio
    async def test_get_model_registry_stats(
        self, lifecycle_service, mock_repositories, sample_model, sample_model_version
    ):
        """Test model registry statistics."""
        _, model_repo, model_version_repo = mock_repositories

        # Mock repositories
        model_repo.find_all.return_value = [sample_model]
        model_version_repo.find_all.return_value = [sample_model_version]

        # Get stats
        stats = await lifecycle_service.get_model_registry_stats()

        # Verify stats
        assert stats["total_models"] == 1
        assert stats["total_versions"] == 1
        assert "model_status_distribution" in stats
        assert "version_status_distribution" in stats
        assert "registry_health" in stats


class TestValidation:
    """Test validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_performance_baseline(
        self, lifecycle_service, sample_model_version
    ):
        """Test performance baseline validation."""
        # Test passing validation
        result = await lifecycle_service._validate_performance_baseline(
            sample_model_version
        )
        assert result["passed"] is True

        # Test failing validation (modify performance)
        sample_model_version.performance_metrics.accuracy = 0.5  # Below threshold
        result = await lifecycle_service._validate_performance_baseline(
            sample_model_version
        )
        assert result["passed"] is False
        assert "accuracy" in result["reason"]

    @pytest.mark.asyncio
    async def test_validate_resource_usage(
        self, lifecycle_service, sample_model_version
    ):
        """Test resource usage validation."""
        # Test passing validation
        result = await lifecycle_service._validate_resource_usage(sample_model_version)
        assert result["passed"] is True

        # Test failing validation (high inference time)
        sample_model_version.performance_metrics.inference_time = 5000  # Too high
        result = await lifecycle_service._validate_resource_usage(sample_model_version)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_run_validation_tests(self, lifecycle_service, sample_model_version):
        """Test running multiple validation tests."""
        validation_tests = ["performance_baseline", "data_drift", "resource_usage"]

        results = await lifecycle_service._run_validation_tests(
            sample_model_version, validation_tests
        )

        # Verify all tests were run
        assert len(results) == len(validation_tests)
        for test_name in validation_tests:
            assert test_name in results
            assert "passed" in results[test_name]


class TestUtilities:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_calculate_checksum(self, lifecycle_service, temp_directories):
        """Test checksum calculation."""
        artifact_path, _ = temp_directories

        # Create test file
        test_file = artifact_path / "test.txt"
        test_file.write_text("test content")

        # Calculate checksum
        checksum = await lifecycle_service._calculate_checksum(test_file)

        # Verify checksum
        assert checksum is not None
        assert len(checksum) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_capture_environment(self, lifecycle_service):
        """Test environment capture."""
        env_info = await lifecycle_service._capture_environment()

        # Verify environment info
        assert "python_version" in env_info
        assert "platform" in env_info
        assert "captured_at" in env_info

    @pytest.mark.asyncio
    async def test_capture_system_info(self, lifecycle_service):
        """Test system info capture."""
        with (
            patch("psutil.cpu_count", return_value=4),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):

            mock_memory.return_value.total = 8 * 1024**3  # 8GB
            mock_disk.return_value.total = 500 * 1024**3  # 500GB

            system_info = await lifecycle_service._capture_system_info()

            # Verify system info
            assert system_info["cpu_count"] == 4
            assert system_info["memory_total"] == 8 * 1024**3
            assert "captured_at" in system_info


class TestIntegration:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_ml_workflow(
        self, lifecycle_service, mock_repositories, temp_directories
    ):
        """Test complete ML workflow from experiment to model registry."""
        experiment_repo, model_repo, model_version_repo = mock_repositories
        artifact_path, _ = temp_directories

        # Mock repositories
        experiment_repo.save = AsyncMock()
        experiment_repo.find_by_id = AsyncMock()
        model_repo.find_by_name.return_value = []
        model_repo.save = AsyncMock()
        model_version_repo.save = AsyncMock()

        # 1. Start experiment
        experiment_id = await lifecycle_service.start_experiment(
            name="Integration Test",
            description="Complete workflow test",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
            objective="Test complete workflow",
            created_by="test_user",
        )

        # Setup mock experiment for run creation
        mock_experiment = MagicMock()
        mock_experiment.status = ExperimentStatus.RUNNING
        mock_experiment.add_run = MagicMock()
        experiment_repo.find_by_id.return_value = mock_experiment

        # 2. Start run
        run_id = await lifecycle_service.start_run(
            experiment_id=experiment_id,
            run_name="Test Run",
            detector_id=uuid4(),
            dataset_id=uuid4(),
            parameters={"contamination": 0.1},
            created_by="test_user",
        )

        # 3. Log parameters and metrics
        await lifecycle_service.log_parameter(run_id, "n_estimators", 100)
        await lifecycle_service.log_metric(run_id, "accuracy", 0.95)

        # 4. Log model
        model = IsolationForest(contamination=0.1)
        X = np.random.random((100, 5))
        model.fit(X)

        model_path = await lifecycle_service.log_model(
            run_id=run_id,
            model=model,
            model_name="test_model",
        )

        # 5. End run
        await lifecycle_service.end_run(run_id, "FINISHED")

        # 6. Create model version
        version_id = await lifecycle_service.create_model_version(
            model_name="test_model_registry",
            run_id=run_id,
            model_path=model_path,
            performance_metrics={"accuracy": 0.95, "f1_score": 0.92},
        )

        # Verify workflow completion
        assert experiment_id is not None
        assert run_id not in lifecycle_service._active_runs  # Run ended
        assert version_id is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_experiment_for_run(
        self, lifecycle_service, mock_repositories
    ):
        """Test error handling for invalid experiment."""
        experiment_repo, _, _ = mock_repositories
        experiment_repo.find_by_id.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await lifecycle_service.start_run(
                experiment_id=str(uuid4()),
                run_name="Test Run",
                detector_id=uuid4(),
                dataset_id=uuid4(),
                parameters={},
                created_by="test_user",
            )

    @pytest.mark.asyncio
    async def test_inactive_run_operations(self, lifecycle_service):
        """Test error handling for inactive run operations."""
        run_id = str(uuid4())

        with pytest.raises(ValueError, match="not found or not active"):
            await lifecycle_service.log_parameter(run_id, "test", "value")

        with pytest.raises(ValueError, match="not found or not active"):
            await lifecycle_service.log_metric(run_id, "test", 0.5)

    @pytest.mark.asyncio
    async def test_invalid_model_version_promotion(
        self, lifecycle_service, mock_repositories
    ):
        """Test error handling for invalid model version promotion."""
        _, _, model_version_repo = mock_repositories
        model_version_repo.find_by_id.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await lifecycle_service.promote_model_version(
                model_version_id=str(uuid4()),
                stage="production",
                promoted_by="test_user",
            )


@pytest.mark.asyncio
async def test_performance_trends_calculation(lifecycle_service, mock_repositories):
    """Test performance trends calculation."""
    _, _, model_version_repo = mock_repositories

    # Create mock versions with different performance
    model_id = uuid4()
    versions = []

    for i in range(3):
        version = MagicMock()
        version.model_id = model_id
        version.created_at = datetime.utcnow() - timedelta(days=i)
        version.get_performance_summary.return_value = {
            "accuracy": 0.8 + (i * 0.05),  # Improving trend
            "f1_score": 0.75 + (i * 0.03),
        }
        versions.append(version)

    # Calculate trends
    trends = await lifecycle_service._calculate_performance_trends(versions)

    # Verify trends
    model_trends = trends[str(model_id)]
    assert model_trends["accuracy"]["trend"] == "improving"
    assert model_trends["f1_score"]["trend"] == "improving"
