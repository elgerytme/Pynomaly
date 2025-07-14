"""
Comprehensive tests for experiment DTOs.

This module tests all experiment-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including experiment creation, run tracking,
leaderboards, and experiment responses.
"""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.experiment_dto import (
    CreateExperimentDTO,
    ExperimentDTO,
    ExperimentResponseDTO,
    LeaderboardEntryDTO,
    RunDTO,
)


class TestRunDTO:
    """Test suite for RunDTO."""

    def test_basic_creation(self):
        """Test basic run DTO creation."""
        run_id = str(uuid4())
        timestamp = datetime.utcnow()
        parameters = {"n_estimators": 100, "contamination": 0.1}
        metrics = {"f1_score": 0.85, "precision": 0.88, "recall": 0.82}

        run = RunDTO(
            id=run_id,
            detector_name="isolation_forest",
            dataset_name="credit_fraud",
            parameters=parameters,
            metrics=metrics,
            timestamp=timestamp,
        )

        assert run.id == run_id
        assert run.detector_name == "isolation_forest"
        assert run.dataset_name == "credit_fraud"
        assert run.parameters == parameters
        assert run.metrics == metrics
        assert run.artifacts == {}  # Default
        assert run.timestamp == timestamp

    def test_creation_with_artifacts(self):
        """Test run DTO creation with artifacts."""
        run_id = str(uuid4())
        timestamp = datetime.utcnow()
        parameters = {"algorithm": "one_class_svm", "gamma": "auto"}
        metrics = {"auc_score": 0.92, "accuracy": 0.89}
        artifacts = {
            "model_path": "/models/run_123/model.pkl",
            "confusion_matrix": "/models/run_123/confusion_matrix.png",
            "feature_importance": "/models/run_123/feature_importance.json",
        }

        run = RunDTO(
            id=run_id,
            detector_name="one_class_svm",
            dataset_name="network_intrusion",
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
            timestamp=timestamp,
        )

        assert run.artifacts == artifacts
        assert len(run.artifacts) == 3

    def test_empty_parameters_and_metrics(self):
        """Test run DTO with empty parameters and metrics."""
        run_id = str(uuid4())
        timestamp = datetime.utcnow()

        run = RunDTO(
            id=run_id,
            detector_name="default_detector",
            dataset_name="test_dataset",
            parameters={},
            metrics={},
            timestamp=timestamp,
        )

        assert run.parameters == {}
        assert run.metrics == {}

    def test_complex_parameters_and_metrics(self):
        """Test run DTO with complex parameters and simple metrics."""
        run_id = str(uuid4())
        timestamp = datetime.utcnow()

        complex_parameters = {
            "ensemble_config": {
                "algorithms": ["isolation_forest", "one_class_svm"],
                "voting_strategy": "weighted_average",
                "weights": [0.6, 0.4],
            },
            "preprocessing": {
                "scaling": "standard",
                "feature_selection": True,
                "n_features": 10,
            },
            "random_state": 42,
        }

        # Metrics must be simple float values per the DTO definition
        metrics = {
            "f1_score": 0.842,
            "precision": 0.881,
            "recall": 0.798,
            "auc_score": 0.912,
            "accuracy": 0.856,
            "cv_mean_f1": 0.838,
            "cv_std_f1": 0.019,
        }

        run = RunDTO(
            id=run_id,
            detector_name="ensemble_detector",
            dataset_name="complex_dataset",
            parameters=complex_parameters,
            metrics=metrics,
            timestamp=timestamp,
        )

        assert (
            run.parameters["ensemble_config"]["voting_strategy"] == "weighted_average"
        )
        assert run.metrics["f1_score"] == 0.842
        assert run.metrics["cv_mean_f1"] == 0.838
        assert len(run.metrics) == 7

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            RunDTO(
                id="test_id",
                detector_name="test",
                dataset_name="test",
                parameters={},
                metrics={},
                timestamp=datetime.utcnow(),
                extra_field="not_allowed",  # type: ignore
            )


class TestExperimentDTO:
    """Test suite for ExperimentDTO."""

    def test_basic_creation(self):
        """Test basic experiment DTO creation."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

        experiment = ExperimentDTO(
            id=experiment_id,
            name="fraud_detection_experiment",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert experiment.id == experiment_id
        assert experiment.name == "fraud_detection_experiment"
        assert experiment.description is None  # Default
        assert experiment.runs == []  # Default
        assert experiment.created_at == created_at
        assert experiment.updated_at == updated_at
        assert experiment.metadata == {}  # Default

    def test_complete_creation(self):
        """Test experiment DTO creation with all fields."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

        # Create some runs
        runs = [
            RunDTO(
                id=str(uuid4()),
                detector_name="isolation_forest",
                dataset_name="credit_fraud",
                parameters={"n_estimators": 100},
                metrics={"f1_score": 0.85},
                timestamp=datetime.utcnow(),
            ),
            RunDTO(
                id=str(uuid4()),
                detector_name="one_class_svm",
                dataset_name="credit_fraud",
                parameters={"gamma": "auto"},
                metrics={"f1_score": 0.82},
                timestamp=datetime.utcnow(),
            ),
        ]

        metadata = {
            "project": "anomaly_detection",
            "owner": "data_science_team",
            "environment": "production",
        }

        experiment = ExperimentDTO(
            id=experiment_id,
            name="credit_fraud_detection",
            description="Comprehensive fraud detection experiment",
            runs=runs,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert experiment.description == "Comprehensive fraud detection experiment"
        assert len(experiment.runs) == 2
        assert experiment.runs[0].detector_name == "isolation_forest"
        assert experiment.runs[1].detector_name == "one_class_svm"
        assert experiment.metadata == metadata

    def test_experiment_with_many_runs(self):
        """Test experiment with many runs."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

        # Create multiple runs for different algorithms
        runs = []
        algorithms = [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope",
        ]

        for i, algorithm in enumerate(algorithms):
            for trial in range(3):  # 3 trials per algorithm
                run = RunDTO(
                    id=str(uuid4()),
                    detector_name=algorithm,
                    dataset_name="benchmark_dataset",
                    parameters={"trial": trial, "random_state": i * 10 + trial},
                    metrics={"f1_score": 0.8 + (i * 0.02) + (trial * 0.01)},
                    timestamp=datetime.utcnow(),
                )
                runs.append(run)

        experiment = ExperimentDTO(
            id=experiment_id,
            name="algorithm_comparison",
            description="Comparison of different anomaly detection algorithms",
            runs=runs,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert len(experiment.runs) == 12  # 4 algorithms * 3 trials
        # Check that we have runs for all algorithms
        algorithm_names = {run.detector_name for run in experiment.runs}
        assert algorithm_names == set(algorithms)

    def test_experiment_evolution(self):
        """Test experiment evolution with runs added over time."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()

        # Initial experiment with no runs
        initial_experiment = ExperimentDTO(
            id=experiment_id,
            name="evolving_experiment",
            created_at=created_at,
            updated_at=created_at,
        )
        assert len(initial_experiment.runs) == 0

        # Add first run
        first_run = RunDTO(
            id=str(uuid4()),
            detector_name="baseline_detector",
            dataset_name="test_data",
            parameters={"baseline": True},
            metrics={"f1_score": 0.75},
            timestamp=datetime.utcnow(),
        )

        updated_experiment = ExperimentDTO(
            id=experiment_id,
            name="evolving_experiment",
            runs=[first_run],
            created_at=created_at,
            updated_at=datetime.utcnow(),
        )
        assert len(updated_experiment.runs) == 1

        # Add more runs
        additional_runs = [
            RunDTO(
                id=str(uuid4()),
                detector_name="improved_detector",
                dataset_name="test_data",
                parameters={"improvement": "v1"},
                metrics={"f1_score": 0.82},
                timestamp=datetime.utcnow(),
            ),
            RunDTO(
                id=str(uuid4()),
                detector_name="optimized_detector",
                dataset_name="test_data",
                parameters={"optimization": "hyperparameter_tuning"},
                metrics={"f1_score": 0.89},
                timestamp=datetime.utcnow(),
            ),
        ]

        final_experiment = ExperimentDTO(
            id=experiment_id,
            name="evolving_experiment",
            runs=[first_run] + additional_runs,
            created_at=created_at,
            updated_at=datetime.utcnow(),
        )
        assert len(final_experiment.runs) == 3
        # Verify performance improvement over time
        f1_scores = [run.metrics["f1_score"] for run in final_experiment.runs]
        assert f1_scores == [0.75, 0.82, 0.89]  # Improving performance


class TestCreateExperimentDTO:
    """Test suite for CreateExperimentDTO."""

    def test_basic_creation(self):
        """Test basic create experiment DTO creation."""
        create_dto = CreateExperimentDTO(name="new_experiment")

        assert create_dto.name == "new_experiment"
        assert create_dto.description is None  # Default
        assert create_dto.metadata == {}  # Default

    def test_complete_creation(self):
        """Test create experiment DTO creation with all fields."""
        metadata = {
            "priority": "high",
            "tags": ["production", "critical"],
            "budget": 1000,
        }

        create_dto = CreateExperimentDTO(
            name="comprehensive_experiment",
            description="A comprehensive experiment for anomaly detection",
            metadata=metadata,
        )

        assert create_dto.name == "comprehensive_experiment"
        assert (
            create_dto.description == "A comprehensive experiment for anomaly detection"
        )
        assert create_dto.metadata == metadata

    def test_name_validation(self):
        """Test name field validation."""
        # Test minimum length
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="")

        # Test maximum length
        long_name = "x" * 101  # 101 characters
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name=long_name)

        # Test valid lengths
        valid_names = ["a", "x" * 100, "valid_experiment_name"]
        for name in valid_names:
            dto = CreateExperimentDTO(name=name)
            assert dto.name == name

    def test_description_validation(self):
        """Test description field validation."""
        # Test maximum length
        long_description = "x" * 501  # 501 characters
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="test", description=long_description)

        # Test valid description
        valid_description = "x" * 500  # Exactly 500 characters
        dto = CreateExperimentDTO(name="test", description=valid_description)
        assert len(dto.description) == 500

        # Test None description
        dto = CreateExperimentDTO(name="test", description=None)
        assert dto.description is None

    def test_empty_metadata(self):
        """Test with empty metadata."""
        dto = CreateExperimentDTO(name="test_experiment", metadata={})
        assert dto.metadata == {}

    def test_complex_metadata(self):
        """Test with complex metadata."""
        complex_metadata = {
            "project_info": {
                "name": "anomaly_detection_v2",
                "version": "2.1.0",
                "team": "ml_engineering",
            },
            "experiment_config": {
                "max_runs": 100,
                "timeout_hours": 24,
                "auto_stop": True,
            },
            "resources": {"compute_type": "gpu", "memory_gb": 16, "storage_gb": 100},
        }

        dto = CreateExperimentDTO(
            name="advanced_experiment",
            description="Advanced experiment with complex configuration",
            metadata=complex_metadata,
        )

        assert dto.metadata["project_info"]["name"] == "anomaly_detection_v2"
        assert dto.metadata["experiment_config"]["max_runs"] == 100
        assert dto.metadata["resources"]["compute_type"] == "gpu"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateExperimentDTO(
                name="test",
                extra_field="not_allowed",  # type: ignore
            )


class TestLeaderboardEntryDTO:
    """Test suite for LeaderboardEntryDTO."""

    def test_basic_creation(self):
        """Test basic leaderboard entry DTO creation."""
        experiment_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.utcnow()

        entry = LeaderboardEntryDTO(
            rank=1,
            experiment_id=experiment_id,
            run_id=run_id,
            detector_name="isolation_forest",
            dataset_name="fraud_detection",
            score=0.92,
            metric_name="f1_score",
            timestamp=timestamp,
        )

        assert entry.rank == 1
        assert entry.experiment_id == experiment_id
        assert entry.run_id == run_id
        assert entry.detector_name == "isolation_forest"
        assert entry.dataset_name == "fraud_detection"
        assert entry.score == 0.92
        assert entry.metric_name == "f1_score"
        assert entry.parameters == {}  # Default
        assert entry.timestamp == timestamp

    def test_creation_with_parameters(self):
        """Test leaderboard entry creation with parameters."""
        experiment_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.utcnow()
        parameters = {"n_estimators": 200, "contamination": 0.05, "random_state": 42}

        entry = LeaderboardEntryDTO(
            rank=3,
            experiment_id=experiment_id,
            run_id=run_id,
            detector_name="ensemble_detector",
            dataset_name="network_intrusion",
            score=0.88,
            metric_name="auc_score",
            parameters=parameters,
            timestamp=timestamp,
        )

        assert entry.rank == 3
        assert entry.score == 0.88
        assert entry.metric_name == "auc_score"
        assert entry.parameters == parameters

    def test_leaderboard_ranking(self):
        """Test multiple leaderboard entries for ranking."""
        base_timestamp = datetime.utcnow()

        entries = [
            LeaderboardEntryDTO(
                rank=1,
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                detector_name="best_detector",
                dataset_name="benchmark",
                score=0.95,
                metric_name="f1_score",
                timestamp=base_timestamp,
            ),
            LeaderboardEntryDTO(
                rank=2,
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                detector_name="good_detector",
                dataset_name="benchmark",
                score=0.91,
                metric_name="f1_score",
                timestamp=base_timestamp,
            ),
            LeaderboardEntryDTO(
                rank=3,
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                detector_name="baseline_detector",
                dataset_name="benchmark",
                score=0.87,
                metric_name="f1_score",
                timestamp=base_timestamp,
            ),
        ]

        # Verify ranking order
        assert entries[0].rank < entries[1].rank < entries[2].rank
        assert entries[0].score > entries[1].score > entries[2].score

    def test_different_metrics(self):
        """Test leaderboard entries with different metrics."""
        base_id = str(uuid4())
        timestamp = datetime.utcnow()

        metrics_entries = [
            ("f1_score", 0.85),
            ("precision", 0.92),
            ("recall", 0.80),
            ("auc_score", 0.89),
            ("accuracy", 0.87),
        ]

        entries = []
        for i, (metric_name, score) in enumerate(metrics_entries):
            entry = LeaderboardEntryDTO(
                rank=i + 1,
                experiment_id=base_id,
                run_id=str(uuid4()),
                detector_name=f"detector_{metric_name}",
                dataset_name="multi_metric_dataset",
                score=score,
                metric_name=metric_name,
                timestamp=timestamp,
            )
            entries.append(entry)

        # Verify different metrics are captured
        metric_names = {entry.metric_name for entry in entries}
        expected_metrics = {"f1_score", "precision", "recall", "auc_score", "accuracy"}
        assert metric_names == expected_metrics

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            LeaderboardEntryDTO(
                rank=1,
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                detector_name="test",
                dataset_name="test",
                score=0.5,
                metric_name="test",
                timestamp=datetime.utcnow(),
                extra_field="not_allowed",  # type: ignore
            )


class TestExperimentResponseDTO:
    """Test suite for ExperimentResponseDTO."""

    def test_basic_creation(self):
        """Test basic experiment response DTO creation."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

        response = ExperimentResponseDTO(
            id=experiment_id,
            name="response_experiment",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert response.id == experiment_id
        assert response.name == "response_experiment"
        assert response.description is None  # Default
        assert response.status == "active"  # Default
        assert response.total_runs == 0  # Default
        assert response.best_score is None  # Default
        assert response.best_metric is None  # Default
        assert response.created_at == created_at
        assert response.updated_at == updated_at
        assert response.metadata == {}  # Default

    def test_complete_creation(self):
        """Test experiment response DTO creation with all fields."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()
        metadata = {
            "total_compute_hours": 15.5,
            "cost_usd": 45.30,
            "best_run_id": str(uuid4()),
        }

        response = ExperimentResponseDTO(
            id=experiment_id,
            name="comprehensive_response",
            description="Complete experiment response",
            status="completed",
            total_runs=25,
            best_score=0.94,
            best_metric="f1_score",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert response.description == "Complete experiment response"
        assert response.status == "completed"
        assert response.total_runs == 25
        assert response.best_score == 0.94
        assert response.best_metric == "f1_score"
        assert response.metadata == metadata

    def test_experiment_status_values(self):
        """Test different experiment status values."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

        statuses = ["active", "completed", "failed", "paused", "cancelled"]

        for status in statuses:
            response = ExperimentResponseDTO(
                id=experiment_id,
                name="status_test",
                status=status,
                created_at=created_at,
                updated_at=updated_at,
            )
            assert response.status == status

    def test_experiment_progression(self):
        """Test experiment progression through different states."""
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()

        # Initial state
        initial_response = ExperimentResponseDTO(
            id=experiment_id,
            name="progression_experiment",
            status="active",
            total_runs=0,
            created_at=created_at,
            updated_at=created_at,
        )
        assert initial_response.total_runs == 0
        assert initial_response.best_score is None

        # In progress state
        progress_response = ExperimentResponseDTO(
            id=experiment_id,
            name="progression_experiment",
            status="active",
            total_runs=5,
            best_score=0.82,
            best_metric="f1_score",
            created_at=created_at,
            updated_at=datetime.utcnow(),
        )
        assert progress_response.total_runs == 5
        assert progress_response.best_score == 0.82

        # Completed state
        completed_response = ExperimentResponseDTO(
            id=experiment_id,
            name="progression_experiment",
            status="completed",
            total_runs=15,
            best_score=0.91,
            best_metric="f1_score",
            created_at=created_at,
            updated_at=datetime.utcnow(),
            metadata={"completion_reason": "max_runs_reached"},
        )
        assert completed_response.status == "completed"
        assert completed_response.total_runs == 15
        assert completed_response.best_score == 0.91
        assert completed_response.metadata["completion_reason"] == "max_runs_reached"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExperimentResponseDTO(
                id=str(uuid4()),
                name="test",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                extra_field="not_allowed",  # type: ignore
            )


class TestExperimentDTOIntegration:
    """Integration tests for experiment DTOs."""

    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow using multiple DTOs."""
        # Step 1: Create experiment
        create_request = CreateExperimentDTO(
            name="fraud_detection_comparison",
            description="Comprehensive comparison of fraud detection algorithms",
            metadata={
                "project": "financial_security",
                "budget": 5000,
                "priority": "high",
            },
        )

        # Step 2: Simulate experiment creation
        experiment_id = str(uuid4())
        created_at = datetime.utcnow()

        experiment = ExperimentDTO(
            id=experiment_id,
            name=create_request.name,
            description=create_request.description,
            created_at=created_at,
            updated_at=created_at,
            metadata=create_request.metadata,
        )

        # Step 3: Add runs to experiment
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        runs = []

        for i, algorithm in enumerate(algorithms):
            run = RunDTO(
                id=str(uuid4()),
                detector_name=algorithm,
                dataset_name="credit_card_fraud",
                parameters={
                    "random_state": 42,
                    "contamination": 0.1,
                    "algorithm_specific": f"param_{i}",
                },
                metrics={
                    "f1_score": 0.8 + (i * 0.03),
                    "precision": 0.85 + (i * 0.02),
                    "recall": 0.78 + (i * 0.04),
                },
                artifacts={
                    "model_path": f"/models/{algorithm}/model.pkl",
                    "report_path": f"/reports/{algorithm}/report.html",
                },
                timestamp=datetime.utcnow(),
            )
            runs.append(run)

        # Update experiment with runs
        updated_experiment = ExperimentDTO(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            runs=runs,
            created_at=experiment.created_at,
            updated_at=datetime.utcnow(),
            metadata=experiment.metadata,
        )

        # Step 4: Create leaderboard entries
        leaderboard_entries = []
        for i, run in enumerate(runs):
            entry = LeaderboardEntryDTO(
                rank=i + 1,
                experiment_id=experiment.id,
                run_id=run.id,
                detector_name=run.detector_name,
                dataset_name=run.dataset_name,
                score=run.metrics["f1_score"],
                metric_name="f1_score",
                parameters=run.parameters,
                timestamp=run.timestamp,
            )
            leaderboard_entries.append(entry)

        # Step 5: Create final response
        best_run = max(runs, key=lambda r: r.metrics["f1_score"])

        response = ExperimentResponseDTO(
            id=updated_experiment.id,
            name=updated_experiment.name,
            description=updated_experiment.description,
            status="completed",
            total_runs=len(runs),
            best_score=best_run.metrics["f1_score"],
            best_metric="f1_score",
            created_at=updated_experiment.created_at,
            updated_at=updated_experiment.updated_at,
            metadata={
                **updated_experiment.metadata,
                "best_algorithm": best_run.detector_name,
                "total_experiments": len(runs),
            },
        )

        # Verify workflow consistency
        assert response.name == create_request.name
        assert response.description == create_request.description
        assert response.total_runs == 3
        assert (
            abs(response.best_score - 0.86) < 0.001
        )  # local_outlier_factor with highest score (with floating point tolerance)
        assert response.metadata["best_algorithm"] == "local_outlier_factor"
        assert len(leaderboard_entries) == 3
        assert leaderboard_entries[0].rank == 1  # Best performer first

    def test_experiment_comparison_analysis(self):
        """Test analysis comparing multiple experiments."""
        base_timestamp = datetime.utcnow()

        # Create multiple experiments for comparison
        experiments = []
        for i in range(3):
            experiment_id = str(uuid4())

            # Create runs for each experiment
            runs = []
            for j in range(2):  # 2 runs per experiment
                run = RunDTO(
                    id=str(uuid4()),
                    detector_name=f"algorithm_{j}",
                    dataset_name=f"dataset_{i}",
                    parameters={"experiment": i, "run": j},
                    metrics={"f1_score": 0.75 + (i * 0.05) + (j * 0.02)},
                    timestamp=base_timestamp,
                )
                runs.append(run)

            experiment = ExperimentDTO(
                id=experiment_id,
                name=f"experiment_{i}",
                description=f"Description for experiment {i}",
                runs=runs,
                created_at=base_timestamp,
                updated_at=base_timestamp,
            )
            experiments.append(experiment)

        # Create responses for comparison
        responses = []
        for experiment in experiments:
            best_run = max(experiment.runs, key=lambda r: r.metrics["f1_score"])

            response = ExperimentResponseDTO(
                id=experiment.id,
                name=experiment.name,
                description=experiment.description,
                status="completed",
                total_runs=len(experiment.runs),
                best_score=best_run.metrics["f1_score"],
                best_metric="f1_score",
                created_at=experiment.created_at,
                updated_at=experiment.updated_at,
            )
            responses.append(response)

        # Verify comparison data
        assert len(responses) == 3
        best_scores = [response.best_score for response in responses]
        expected_scores = [0.77, 0.82, 0.87]
        for actual, expected in zip(best_scores, expected_scores, strict=False):
            assert (
                abs(actual - expected) < 0.001
            )  # Improving scores with floating point tolerance

        # Find overall best experiment
        best_experiment = max(responses, key=lambda r: r.best_score)
        assert best_experiment.name == "experiment_2"
        assert abs(best_experiment.best_score - 0.87) < 0.001

    def test_leaderboard_generation(self):
        """Test leaderboard generation from multiple experiments."""
        # Create runs from different experiments
        runs_data = [
            ("exp_1", "run_1", "isolation_forest", "dataset_a", 0.89),
            ("exp_1", "run_2", "one_class_svm", "dataset_a", 0.85),
            ("exp_2", "run_3", "local_outlier_factor", "dataset_b", 0.92),
            ("exp_2", "run_4", "ensemble", "dataset_b", 0.94),
            ("exp_3", "run_5", "isolation_forest", "dataset_c", 0.87),
        ]

        leaderboard_entries = []
        for i, (exp_id, run_id, detector, dataset, score) in enumerate(runs_data):
            entry = LeaderboardEntryDTO(
                rank=i + 1,  # Will be reordered by score later
                experiment_id=exp_id,
                run_id=run_id,
                detector_name=detector,
                dataset_name=dataset,
                score=score,
                metric_name="f1_score",
                timestamp=datetime.utcnow(),
            )
            leaderboard_entries.append(entry)

        # Sort by score (descending) and update ranks
        sorted_entries = sorted(
            leaderboard_entries, key=lambda x: x.score, reverse=True
        )
        for i, entry in enumerate(sorted_entries):
            entry.rank = i + 1

        # Verify leaderboard order
        assert sorted_entries[0].detector_name == "ensemble"
        assert sorted_entries[0].score == 0.94
        assert sorted_entries[0].rank == 1

        assert sorted_entries[1].detector_name == "local_outlier_factor"
        assert sorted_entries[1].score == 0.92
        assert sorted_entries[1].rank == 2

        assert sorted_entries[-1].detector_name == "one_class_svm"
        assert sorted_entries[-1].score == 0.85
        assert sorted_entries[-1].rank == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
