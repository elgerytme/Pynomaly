"""
Comprehensive tests for Experiment Tracking Service.
Tests experiment creation, run logging, comparison, and reporting functionality.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pynomaly.application.services.experiment_tracking_service import (
    ExperimentTrackingService,
)


class TestExperimentTrackingService:
    """Test suite for ExperimentTrackingService."""

    @pytest.fixture
    def temp_tracking_path(self):
        """Create temporary tracking directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tracking_service(self, temp_tracking_path):
        """Experiment tracking service with temporary storage."""
        return ExperimentTrackingService(tracking_path=temp_tracking_path)

    @pytest.fixture
    def sample_experiment_data(self):
        """Sample experiment data for testing."""
        return {
            "name": "Anomaly Detection Comparison",
            "description": "Comparing different anomaly detection algorithms",
            "tags": ["anomaly_detection", "comparison", "ml"],
        }

    @pytest.fixture
    def sample_run_data(self):
        """Sample run data for testing."""
        return {
            "detector_name": "IsolationForest",
            "dataset_name": "credit_card_fraud",
            "parameters": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.88,
                "recall": 0.82,
                "f1": 0.85,
                "auc_roc": 0.91,
            },
            "artifacts": {
                "model_file": "/path/to/model.pkl",
                "confusion_matrix": "/path/to/confusion_matrix.png",
            },
        }

    def test_service_initialization(self, temp_tracking_path):
        """Test service initialization and directory creation."""
        service = ExperimentTrackingService(tracking_path=temp_tracking_path)

        assert service.tracking_path == temp_tracking_path
        assert temp_tracking_path.exists()
        assert service.experiments_file == temp_tracking_path / "experiments.json"
        assert isinstance(service.experiments, dict)

    def test_service_initialization_with_existing_experiments(self, temp_tracking_path):
        """Test service initialization with existing experiments file."""
        # Create existing experiments file
        existing_experiments = {
            "exp1": {
                "id": "exp1",
                "name": "Existing Experiment",
                "runs": [],
            }
        }

        experiments_file = temp_tracking_path / "experiments.json"
        with open(experiments_file, "w") as f:
            json.dump(existing_experiments, f)

        service = ExperimentTrackingService(tracking_path=temp_tracking_path)

        assert "exp1" in service.experiments
        assert service.experiments["exp1"]["name"] == "Existing Experiment"

    @pytest.mark.asyncio
    async def test_create_experiment_basic(
        self, tracking_service, sample_experiment_data
    ):
        """Test basic experiment creation."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        assert experiment_id is not None
        assert isinstance(experiment_id, str)
        assert experiment_id in tracking_service.experiments

        experiment = tracking_service.experiments[experiment_id]
        assert experiment["id"] == experiment_id
        assert experiment["name"] == sample_experiment_data["name"]
        assert experiment["description"] == sample_experiment_data["description"]
        assert experiment["tags"] == sample_experiment_data["tags"]
        assert "created_at" in experiment
        assert experiment["runs"] == []

        # Check that experiment directory was created
        exp_dir = tracking_service.tracking_path / experiment_id
        assert exp_dir.exists()

        # Check that experiments file was saved
        assert tracking_service.experiments_file.exists()

    @pytest.mark.asyncio
    async def test_create_experiment_minimal(self, tracking_service):
        """Test experiment creation with minimal data."""
        experiment_id = await tracking_service.create_experiment(
            name="Minimal Experiment"
        )

        assert experiment_id is not None
        experiment = tracking_service.experiments[experiment_id]
        assert experiment["name"] == "Minimal Experiment"
        assert experiment["description"] is None
        assert experiment["tags"] == []

    @pytest.mark.asyncio
    async def test_create_multiple_experiments(self, tracking_service):
        """Test creating multiple experiments."""
        exp1_id = await tracking_service.create_experiment(name="Experiment 1")
        exp2_id = await tracking_service.create_experiment(name="Experiment 2")

        assert exp1_id != exp2_id
        assert len(tracking_service.experiments) == 2
        assert tracking_service.experiments[exp1_id]["name"] == "Experiment 1"
        assert tracking_service.experiments[exp2_id]["name"] == "Experiment 2"

    @pytest.mark.asyncio
    async def test_log_run_basic(
        self, tracking_service, sample_experiment_data, sample_run_data
    ):
        """Test basic run logging."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run_id = await tracking_service.log_run(experiment_id, **sample_run_data)

        assert run_id is not None
        assert isinstance(run_id, str)

        experiment = tracking_service.experiments[experiment_id]
        assert len(experiment["runs"]) == 1

        run = experiment["runs"][0]
        assert run["id"] == run_id
        assert run["detector_name"] == sample_run_data["detector_name"]
        assert run["dataset_name"] == sample_run_data["dataset_name"]
        assert run["parameters"] == sample_run_data["parameters"]
        assert run["metrics"] == sample_run_data["metrics"]
        assert run["artifacts"] == sample_run_data["artifacts"]
        assert "timestamp" in run

        # Check that run file was created
        run_dir = tracking_service.tracking_path / experiment_id / run_id
        assert run_dir.exists()
        run_file = run_dir / "run.json"
        assert run_file.exists()

        # Verify run file content
        with open(run_file) as f:
            saved_run = json.load(f)
        assert saved_run["id"] == run_id

    @pytest.mark.asyncio
    async def test_log_run_without_artifacts(
        self, tracking_service, sample_experiment_data
    ):
        """Test logging run without artifacts."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run_data = {
            "detector_name": "LOF",
            "dataset_name": "test_data",
            "parameters": {"n_neighbors": 20},
            "metrics": {"f1": 0.75},
        }

        run_id = await tracking_service.log_run(experiment_id, **run_data)

        experiment = tracking_service.experiments[experiment_id]
        run = experiment["runs"][0]
        assert run["artifacts"] == {}

    @pytest.mark.asyncio
    async def test_log_run_nonexistent_experiment(
        self, tracking_service, sample_run_data
    ):
        """Test logging run to nonexistent experiment."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await tracking_service.log_run("nonexistent_id", **sample_run_data)

    @pytest.mark.asyncio
    async def test_log_multiple_runs(self, tracking_service, sample_experiment_data):
        """Test logging multiple runs to the same experiment."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run_data_1 = {
            "detector_name": "IsolationForest",
            "dataset_name": "dataset1",
            "parameters": {"n_estimators": 100},
            "metrics": {"f1": 0.85},
        }

        run_data_2 = {
            "detector_name": "LOF",
            "dataset_name": "dataset1",
            "parameters": {"n_neighbors": 20},
            "metrics": {"f1": 0.80},
        }

        run1_id = await tracking_service.log_run(experiment_id, **run_data_1)
        run2_id = await tracking_service.log_run(experiment_id, **run_data_2)

        assert run1_id != run2_id
        experiment = tracking_service.experiments[experiment_id]
        assert len(experiment["runs"]) == 2

    @pytest.mark.asyncio
    async def test_compare_runs_all_runs(
        self, tracking_service, sample_experiment_data
    ):
        """Test comparing all runs in an experiment."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        # Log multiple runs
        runs_data = [
            {
                "detector_name": "IsolationForest",
                "dataset_name": "test_data",
                "parameters": {"n_estimators": 100},
                "metrics": {"f1": 0.85, "precision": 0.88},
            },
            {
                "detector_name": "LOF",
                "dataset_name": "test_data",
                "parameters": {"n_neighbors": 20},
                "metrics": {"f1": 0.80, "precision": 0.82},
            },
            {
                "detector_name": "OneClassSVM",
                "dataset_name": "test_data",
                "parameters": {"nu": 0.1},
                "metrics": {"f1": 0.75, "precision": 0.79},
            },
        ]

        for run_data in runs_data:
            await tracking_service.log_run(experiment_id, **run_data)

        comparison_df = await tracking_service.compare_runs(experiment_id)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert "run_id" in comparison_df.columns
        assert "detector" in comparison_df.columns
        assert "dataset" in comparison_df.columns
        assert "f1" in comparison_df.columns
        assert "precision" in comparison_df.columns

        # Should be sorted by f1 score (default metric)
        assert comparison_df.iloc[0]["f1"] >= comparison_df.iloc[1]["f1"]
        assert comparison_df.iloc[1]["f1"] >= comparison_df.iloc[2]["f1"]

        # Check parameter columns
        assert "param_n_estimators" in comparison_df.columns or pd.isna(
            comparison_df.iloc[1]["param_n_estimators"]
        )

    @pytest.mark.asyncio
    async def test_compare_runs_specific_runs(
        self, tracking_service, sample_experiment_data
    ):
        """Test comparing specific runs."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run1_id = await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={"n_estimators": 100},
            metrics={"f1": 0.85},
        )

        run2_id = await tracking_service.log_run(
            experiment_id,
            detector_name="LOF",
            dataset_name="test_data",
            parameters={"n_neighbors": 20},
            metrics={"f1": 0.80},
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="OneClassSVM",
            dataset_name="test_data",
            parameters={"nu": 0.1},
            metrics={"f1": 0.75},
        )

        # Compare only first two runs
        comparison_df = await tracking_service.compare_runs(
            experiment_id, run_ids=[run1_id, run2_id]
        )

        assert len(comparison_df) == 2
        assert run1_id in comparison_df["run_id"].values
        assert run2_id in comparison_df["run_id"].values

    @pytest.mark.asyncio
    async def test_compare_runs_custom_metric(
        self, tracking_service, sample_experiment_data
    ):
        """Test comparing runs with custom metric."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.75, "precision": 0.90},
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="LOF",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.85, "precision": 0.80},
        )

        comparison_df = await tracking_service.compare_runs(
            experiment_id, metric="precision"
        )

        # Should be sorted by precision
        assert comparison_df.iloc[0]["precision"] >= comparison_df.iloc[1]["precision"]

    @pytest.mark.asyncio
    async def test_compare_runs_nonexistent_experiment(self, tracking_service):
        """Test comparing runs for nonexistent experiment."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await tracking_service.compare_runs("nonexistent_id")

    @pytest.mark.asyncio
    async def test_get_best_run_basic(self, tracking_service, sample_experiment_data):
        """Test getting best run from experiment."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        runs_data = [
            {"detector_name": "IsolationForest", "metrics": {"f1": 0.85}},
            {"detector_name": "LOF", "metrics": {"f1": 0.90}},
            {"detector_name": "OneClassSVM", "metrics": {"f1": 0.75}},
        ]

        run_ids = []
        for run_data in runs_data:
            run_id = await tracking_service.log_run(
                experiment_id,
                dataset_name="test_data",
                parameters={},
                **run_data,
            )
            run_ids.append(run_id)

        best_run = await tracking_service.get_best_run(experiment_id)

        assert best_run is not None
        assert best_run["detector_name"] == "LOF"
        assert best_run["metrics"]["f1"] == 0.90

    @pytest.mark.asyncio
    async def test_get_best_run_custom_metric(
        self, tracking_service, sample_experiment_data
    ):
        """Test getting best run with custom metric."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.85, "precision": 0.80},
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="LOF",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.75, "precision": 0.95},
        )

        best_run = await tracking_service.get_best_run(
            experiment_id, metric="precision"
        )

        assert best_run["detector_name"] == "LOF"
        assert best_run["metrics"]["precision"] == 0.95

    @pytest.mark.asyncio
    async def test_get_best_run_lower_is_better(
        self, tracking_service, sample_experiment_data
    ):
        """Test getting best run where lower values are better."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={},
            metrics={"error_rate": 0.15},
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="LOF",
            dataset_name="test_data",
            parameters={},
            metrics={"error_rate": 0.10},
        )

        best_run = await tracking_service.get_best_run(
            experiment_id, metric="error_rate", higher_is_better=False
        )

        assert best_run["detector_name"] == "LOF"
        assert best_run["metrics"]["error_rate"] == 0.10

    @pytest.mark.asyncio
    async def test_get_best_run_no_runs(self, tracking_service, sample_experiment_data):
        """Test getting best run when no runs exist."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        with pytest.raises(ValueError, match="No runs found in experiment"):
            await tracking_service.get_best_run(experiment_id)

    @pytest.mark.asyncio
    async def test_get_best_run_no_metric(
        self, tracking_service, sample_experiment_data
    ):
        """Test getting best run when metric doesn't exist."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.85},
        )

        with pytest.raises(ValueError, match="No runs found with metric"):
            await tracking_service.get_best_run(
                experiment_id, metric="nonexistent_metric"
            )

    @pytest.mark.asyncio
    async def test_log_artifact(self, tracking_service, sample_experiment_data):
        """Test logging artifacts for a run."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run_id = await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={},
            metrics={"f1": 0.85},
        )

        await tracking_service.log_artifact(
            experiment_id, run_id, "model", "/path/to/model.pkl"
        )

        experiment = tracking_service.experiments[experiment_id]
        run = experiment["runs"][0]
        assert "model" in run["artifacts"]
        assert run["artifacts"]["model"] == "/path/to/model.pkl"

    @pytest.mark.asyncio
    async def test_log_artifact_nonexistent_experiment(self, tracking_service):
        """Test logging artifact for nonexistent experiment."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await tracking_service.log_artifact(
                "nonexistent_exp", "run_id", "artifact", "path"
            )

    @pytest.mark.asyncio
    async def test_log_artifact_nonexistent_run(
        self, tracking_service, sample_experiment_data
    ):
        """Test logging artifact for nonexistent run."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        with pytest.raises(ValueError, match="Run .* not found"):
            await tracking_service.log_artifact(
                experiment_id, "nonexistent_run", "artifact", "path"
            )

    @pytest.mark.asyncio
    async def test_create_leaderboard_all_experiments(self, tracking_service):
        """Test creating leaderboard across all experiments."""
        # Create multiple experiments with runs
        exp1_id = await tracking_service.create_experiment(name="Experiment 1")
        exp2_id = await tracking_service.create_experiment(name="Experiment 2")

        # Add runs to first experiment
        await tracking_service.log_run(
            exp1_id,
            detector_name="IsolationForest",
            dataset_name="dataset1",
            parameters={},
            metrics={"f1": 0.85},
        )

        await tracking_service.log_run(
            exp1_id,
            detector_name="LOF",
            dataset_name="dataset1",
            parameters={},
            metrics={"f1": 0.80},
        )

        # Add runs to second experiment
        await tracking_service.log_run(
            exp2_id,
            detector_name="OneClassSVM",
            dataset_name="dataset2",
            parameters={},
            metrics={"f1": 0.90},
        )

        leaderboard = await tracking_service.create_leaderboard()

        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 3
        assert "experiment" in leaderboard.columns
        assert "run_id" in leaderboard.columns
        assert "detector" in leaderboard.columns
        assert "dataset" in leaderboard.columns
        assert "f1" in leaderboard.columns
        assert "rank" in leaderboard.columns

        # Should be sorted by f1 score
        assert leaderboard.iloc[0]["f1"] == 0.90
        assert leaderboard.iloc[0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_create_leaderboard_specific_experiments(self, tracking_service):
        """Test creating leaderboard for specific experiments."""
        exp1_id = await tracking_service.create_experiment(name="Experiment 1")
        exp2_id = await tracking_service.create_experiment(name="Experiment 2")
        exp3_id = await tracking_service.create_experiment(name="Experiment 3")

        # Add runs
        await tracking_service.log_run(
            exp1_id,
            detector_name="IsolationForest",
            dataset_name="dataset1",
            parameters={},
            metrics={"f1": 0.85},
        )

        await tracking_service.log_run(
            exp2_id,
            detector_name="LOF",
            dataset_name="dataset2",
            parameters={},
            metrics={"f1": 0.80},
        )

        await tracking_service.log_run(
            exp3_id,
            detector_name="OneClassSVM",
            dataset_name="dataset3",
            parameters={},
            metrics={"f1": 0.90},
        )

        # Create leaderboard for only first two experiments
        leaderboard = await tracking_service.create_leaderboard(
            experiment_ids=[exp1_id, exp2_id]
        )

        assert len(leaderboard) == 2
        assert all(
            exp in ["Experiment 1", "Experiment 2"] for exp in leaderboard["experiment"]
        )

    @pytest.mark.asyncio
    async def test_create_leaderboard_custom_metric(self, tracking_service):
        """Test creating leaderboard with custom metric."""
        exp_id = await tracking_service.create_experiment(name="Test Experiment")

        await tracking_service.log_run(
            exp_id,
            detector_name="IsolationForest",
            dataset_name="dataset1",
            parameters={},
            metrics={"f1": 0.75, "precision": 0.90},
        )

        await tracking_service.log_run(
            exp_id,
            detector_name="LOF",
            dataset_name="dataset1",
            parameters={},
            metrics={"f1": 0.85, "precision": 0.80},
        )

        leaderboard = await tracking_service.create_leaderboard(metric="precision")

        assert leaderboard.iloc[0]["precision"] == 0.90
        assert leaderboard.iloc[0]["detector"] == "IsolationForest"

    @pytest.mark.asyncio
    async def test_create_leaderboard_empty(self, tracking_service):
        """Test creating leaderboard with no experiments."""
        leaderboard = await tracking_service.create_leaderboard()

        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 0

    @pytest.mark.asyncio
    async def test_export_experiment(
        self, tracking_service, temp_tracking_path, sample_experiment_data
    ):
        """Test exporting experiment data."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        # Add some runs
        await tracking_service.log_run(
            experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_data",
            parameters={"n_estimators": 100},
            metrics={"f1": 0.85, "precision": 0.88},
        )

        await tracking_service.log_run(
            experiment_id,
            detector_name="LOF",
            dataset_name="test_data",
            parameters={"n_neighbors": 20},
            metrics={"f1": 0.80, "precision": 0.82},
        )

        export_path = temp_tracking_path / "export"
        await tracking_service.export_experiment(experiment_id, export_path)

        # Check exported files
        assert export_path.exists()
        assert (export_path / "experiment.json").exists()
        assert (export_path / "comparison.csv").exists()
        assert (export_path / "report.md").exists()

        # Verify experiment.json content
        with open(export_path / "experiment.json") as f:
            exported_exp = json.load(f)
        assert exported_exp["name"] == sample_experiment_data["name"]
        assert len(exported_exp["runs"]) == 2

        # Verify comparison.csv content
        comparison_df = pd.read_csv(export_path / "comparison.csv")
        assert len(comparison_df) == 2
        assert "detector" in comparison_df.columns

        # Verify report.md content
        with open(export_path / "report.md") as f:
            report_content = f.read()
        assert sample_experiment_data["name"] in report_content
        assert "Total runs: 2" in report_content

    @pytest.mark.asyncio
    async def test_export_experiment_nonexistent(
        self, tracking_service, temp_tracking_path
    ):
        """Test exporting nonexistent experiment."""
        export_path = temp_tracking_path / "export"

        with pytest.raises(ValueError, match="Experiment .* not found"):
            await tracking_service.export_experiment("nonexistent_id", export_path)

    def test_generate_experiment_report(self, tracking_service):
        """Test experiment report generation."""
        experiment = {
            "id": "test_id",
            "name": "Test Experiment",
            "created_at": datetime.utcnow().isoformat(),
            "description": "Test description",
            "tags": ["test", "example"],
            "runs": [
                {
                    "id": "run1",
                    "detector_name": "IsolationForest",
                    "dataset_name": "test_data",
                    "timestamp": datetime.utcnow().isoformat(),
                    "parameters": {"n_estimators": 100},
                    "metrics": {"f1": 0.85, "precision": 0.88, "auc_roc": 0.91},
                },
                {
                    "id": "run2",
                    "detector_name": "LOF",
                    "dataset_name": "test_data",
                    "timestamp": datetime.utcnow().isoformat(),
                    "parameters": {"n_neighbors": 20},
                    "metrics": {"f1": 0.80, "precision": 0.82, "auc_roc": 0.88},
                },
            ],
        }

        report = tracking_service._generate_experiment_report(experiment)

        assert isinstance(report, str)
        assert "# Experiment: Test Experiment" in report
        assert "test_id" in report
        assert "Test description" in report
        assert "test, example" in report
        assert "Total runs: 2" in report

        # Check best performing runs table
        assert "| f1 | 0.8500 |" in report
        assert "| auc_roc | 0.9100 |" in report

        # Check individual runs section
        assert "### Run 1: IsolationForest" in report
        assert "### Run 2: LOF" in report

    @pytest.mark.asyncio
    async def test_file_persistence(self, temp_tracking_path, sample_experiment_data):
        """Test that data persists across service instances."""
        # Create service and add experiment
        service1 = ExperimentTrackingService(tracking_path=temp_tracking_path)
        exp_id = await service1.create_experiment(**sample_experiment_data)

        # Create new service instance
        service2 = ExperimentTrackingService(tracking_path=temp_tracking_path)

        # Should load existing experiments
        assert exp_id in service2.experiments
        assert service2.experiments[exp_id]["name"] == sample_experiment_data["name"]

    @pytest.mark.asyncio
    async def test_concurrent_modifications(
        self, tracking_service, sample_experiment_data
    ):
        """Test handling of concurrent modifications."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        # Simulate concurrent run logging
        run_data_1 = {
            "detector_name": "IsolationForest",
            "dataset_name": "dataset1",
            "parameters": {"n_estimators": 100},
            "metrics": {"f1": 0.85},
        }

        run_data_2 = {
            "detector_name": "LOF",
            "dataset_name": "dataset1",
            "parameters": {"n_neighbors": 20},
            "metrics": {"f1": 0.80},
        }

        # Both runs should be logged successfully
        run1_id = await tracking_service.log_run(experiment_id, **run_data_1)
        run2_id = await tracking_service.log_run(experiment_id, **run_data_2)

        assert run1_id != run2_id
        experiment = tracking_service.experiments[experiment_id]
        assert len(experiment["runs"]) == 2

    @pytest.mark.asyncio
    async def test_large_experiment_handling(self, tracking_service):
        """Test handling of experiments with many runs."""
        experiment_id = await tracking_service.create_experiment(
            name="Large Experiment"
        )

        # Log many runs
        n_runs = 50
        for i in range(n_runs):
            await tracking_service.log_run(
                experiment_id,
                detector_name=f"Detector_{i % 5}",
                dataset_name=f"Dataset_{i % 3}",
                parameters={"param": i},
                metrics={"f1": 0.5 + (i % 10) * 0.05},
            )

        experiment = tracking_service.experiments[experiment_id]
        assert len(experiment["runs"]) == n_runs

        # Test comparison with many runs
        comparison_df = await tracking_service.compare_runs(experiment_id)
        assert len(comparison_df) == n_runs

        # Test best run selection
        best_run = await tracking_service.get_best_run(experiment_id)
        assert best_run is not None

    def test_load_experiments_corrupted_file(self, temp_tracking_path):
        """Test handling of corrupted experiments file."""
        # Create corrupted experiments file
        experiments_file = temp_tracking_path / "experiments.json"
        with open(experiments_file, "w") as f:
            f.write("corrupted json content")

        # Should handle corrupted file gracefully
        with pytest.raises(json.JSONDecodeError):
            ExperimentTrackingService(tracking_path=temp_tracking_path)

    @pytest.mark.asyncio
    async def test_experiment_with_complex_parameters(self, tracking_service):
        """Test experiment with complex parameter structures."""
        experiment_id = await tracking_service.create_experiment(
            name="Complex Parameters"
        )

        complex_parameters = {
            "model_config": {
                "n_estimators": 100,
                "max_depth": None,
                "nested": {"value": 42, "list": [1, 2, 3]},
            },
            "preprocessing": ["StandardScaler", "PCA"],
            "hyperparameter_grid": {
                "contamination": [0.05, 0.1, 0.15],
                "random_state": 42,
            },
        }

        run_id = await tracking_service.log_run(
            experiment_id,
            detector_name="ComplexDetector",
            dataset_name="test_data",
            parameters=complex_parameters,
            metrics={"f1": 0.85},
        )

        experiment = tracking_service.experiments[experiment_id]
        saved_run = experiment["runs"][0]
        assert saved_run["parameters"] == complex_parameters

    @pytest.mark.asyncio
    async def test_experiment_timestamps(
        self, tracking_service, sample_experiment_data
    ):
        """Test that timestamps are properly recorded."""
        with patch(
            "pynomaly.application.services.experiment_tracking_service.datetime"
        ) as mock_datetime:
            mock_time = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_time

            experiment_id = await tracking_service.create_experiment(
                **sample_experiment_data
            )

            experiment = tracking_service.experiments[experiment_id]
            assert experiment["created_at"] == mock_time.isoformat()

            # Test run timestamp
            run_id = await tracking_service.log_run(
                experiment_id,
                detector_name="Test",
                dataset_name="test_data",
                parameters={},
                metrics={"f1": 0.85},
            )

            run = experiment["runs"][0]
            assert run["timestamp"] == mock_time.isoformat()

    @pytest.mark.asyncio
    async def test_empty_metrics_and_parameters(
        self, tracking_service, sample_experiment_data
    ):
        """Test handling of empty metrics and parameters."""
        experiment_id = await tracking_service.create_experiment(
            **sample_experiment_data
        )

        run_id = await tracking_service.log_run(
            experiment_id,
            detector_name="MinimalDetector",
            dataset_name="test_data",
            parameters={},
            metrics={},
        )

        experiment = tracking_service.experiments[experiment_id]
        run = experiment["runs"][0]
        assert run["parameters"] == {}
        assert run["metrics"] == {}

        # Should handle comparison even with empty metrics
        comparison_df = await tracking_service.compare_runs(experiment_id)
        assert len(comparison_df) == 1
