"""Tests for Experiment DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.experiment_dto import (
    CreateExperimentDTO,
    ExperimentDTO,
    ExperimentResponseDTO,
    LeaderboardEntryDTO,
    RunDTO,
)


class TestRunDTO:
    """Test suite for RunDTO."""

    def test_valid_creation(self):
        """Test creating a valid run DTO."""
        run_id = str(uuid4())
        timestamp = datetime.now()
        parameters = {"n_estimators": 100, "contamination": 0.1}
        metrics = {"accuracy": 0.85, "precision": 0.78, "recall": 0.92}
        artifacts = {
            "model_path": "/path/to/model.pkl",
            "plot_path": "/path/to/plot.png",
        }

        dto = RunDTO(
            id=run_id,
            detector_name="IsolationForest",
            dataset_name="fraud_dataset",
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
            timestamp=timestamp,
        )

        assert dto.id == run_id
        assert dto.detector_name == "IsolationForest"
        assert dto.dataset_name == "fraud_dataset"
        assert dto.parameters == parameters
        assert dto.metrics == metrics
        assert dto.artifacts == artifacts
        assert dto.timestamp == timestamp

    def test_default_artifacts(self):
        """Test default artifacts value."""
        run_id = str(uuid4())
        timestamp = datetime.now()

        dto = RunDTO(
            id=run_id,
            detector_name="OneClassSVM",
            dataset_name="test_dataset",
            parameters={"nu": 0.05},
            metrics={"f1_score": 0.82},
            timestamp=timestamp,
        )

        assert dto.artifacts == {}

    def test_complex_parameters(self):
        """Test complex parameter structures."""
        run_id = str(uuid4())
        timestamp = datetime.now()

        complex_params = {
            "algorithm": "IsolationForest",
            "hyperparameters": {
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": 0.1,
                "random_state": 42,
            },
            "preprocessing": {
                "scaler": "StandardScaler",
                "feature_selection": True,
                "pca_components": 10,
            },
            "cross_validation": {"folds": 5, "strategy": "stratified"},
        }

        dto = RunDTO(
            id=run_id,
            detector_name="IsolationForest",
            dataset_name="complex_dataset",
            parameters=complex_params,
            metrics={"auc": 0.89},
            timestamp=timestamp,
        )

        assert dto.parameters["algorithm"] == "IsolationForest"
        assert dto.parameters["hyperparameters"]["n_estimators"] == 200
        assert dto.parameters["preprocessing"]["scaler"] == "StandardScaler"
        assert dto.parameters["cross_validation"]["folds"] == 5

    def test_complex_metrics(self):
        """Test complex metric structures."""
        run_id = str(uuid4())
        timestamp = datetime.now()

        complex_metrics = {
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.92,
            "f1_score": 0.84,
            "auc_score": 0.91,
            "confusion_matrix": {
                "true_positives": 150,
                "false_positives": 25,
                "true_negatives": 800,
                "false_negatives": 25,
            },
            "cross_validation": {
                "mean_score": 0.83,
                "std_score": 0.05,
                "fold_scores": [0.81, 0.85, 0.82, 0.84, 0.83],
            },
        }

        dto = RunDTO(
            id=run_id,
            detector_name="LocalOutlierFactor",
            dataset_name="validation_dataset",
            parameters={"n_neighbors": 20},
            metrics=complex_metrics,
            timestamp=timestamp,
        )

        assert dto.metrics["accuracy"] == 0.85
        assert dto.metrics["confusion_matrix"]["true_positives"] == 150
        assert dto.metrics["cross_validation"]["mean_score"] == 0.83
        assert len(dto.metrics["cross_validation"]["fold_scores"]) == 5

    def test_artifacts_handling(self):
        """Test artifacts handling."""
        run_id = str(uuid4())
        timestamp = datetime.now()

        artifacts = {
            "model_file": "/experiments/run_123/model.pkl",
            "scaler_file": "/experiments/run_123/scaler.pkl",
            "feature_importance_plot": "/experiments/run_123/feature_importance.png",
            "roc_curve_plot": "/experiments/run_123/roc_curve.png",
            "confusion_matrix_plot": "/experiments/run_123/confusion_matrix.png",
            "training_log": "/experiments/run_123/training.log",
            "config_file": "/experiments/run_123/config.json",
        }

        dto = RunDTO(
            id=run_id,
            detector_name="EllipticEnvelope",
            dataset_name="network_intrusion",
            parameters={"contamination": 0.05},
            metrics={"precision": 0.91},
            artifacts=artifacts,
            timestamp=timestamp,
        )

        assert dto.artifacts["model_file"] == "/experiments/run_123/model.pkl"
        assert dto.artifacts["training_log"] == "/experiments/run_123/training.log"
        assert len(dto.artifacts) == 7

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            RunDTO(
                detector_name="IsolationForest",
                dataset_name="test_dataset",
                parameters={},
                metrics={},
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            RunDTO(
                id=str(uuid4()),
                detector_name="IsolationForest",
                dataset_name="test_dataset",
                parameters={},
                metrics={},
                timestamp=datetime.now(),
                unknown_field="value",
            )


class TestExperimentDTO:
    """Test suite for ExperimentDTO."""

    def test_valid_creation(self):
        """Test creating a valid experiment DTO."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()
        metadata = {"project": "anomaly_detection", "version": "1.0"}

        run1 = RunDTO(
            id=str(uuid4()),
            detector_name="IsolationForest",
            dataset_name="dataset1",
            parameters={"n_estimators": 100},
            metrics={"accuracy": 0.85},
            timestamp=datetime.now(),
        )

        run2 = RunDTO(
            id=str(uuid4()),
            detector_name="OneClassSVM",
            dataset_name="dataset1",
            parameters={"nu": 0.05},
            metrics={"accuracy": 0.82},
            timestamp=datetime.now(),
        )

        dto = ExperimentDTO(
            id=exp_id,
            name="fraud_detection_experiment",
            description="Experiment to detect fraudulent transactions",
            runs=[run1, run2],
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert dto.id == exp_id
        assert dto.name == "fraud_detection_experiment"
        assert dto.description == "Experiment to detect fraudulent transactions"
        assert len(dto.runs) == 2
        assert dto.runs[0] == run1
        assert dto.runs[1] == run2
        assert dto.created_at == created_at
        assert dto.updated_at == updated_at
        assert dto.metadata == metadata

    def test_default_values(self):
        """Test default values."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        dto = ExperimentDTO(
            id=exp_id,
            name="test_experiment",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert dto.description is None
        assert dto.runs == []
        assert dto.metadata == {}

    def test_optional_description(self):
        """Test optional description field."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        # With description
        dto_with_desc = ExperimentDTO(
            id=exp_id,
            name="test_experiment",
            description="Test experiment description",
            created_at=created_at,
            updated_at=updated_at,
        )
        assert dto_with_desc.description == "Test experiment description"

        # Without description
        dto_without_desc = ExperimentDTO(
            id=exp_id,
            name="test_experiment",
            created_at=created_at,
            updated_at=updated_at,
        )
        assert dto_without_desc.description is None

    def test_multiple_runs(self):
        """Test experiment with multiple runs."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        runs = []
        algorithms = [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
            "EllipticEnvelope",
        ]

        for i, algorithm in enumerate(algorithms):
            run = RunDTO(
                id=str(uuid4()),
                detector_name=algorithm,
                dataset_name="benchmark_dataset",
                parameters={"random_state": 42},
                metrics={"accuracy": 0.8 + i * 0.02},
                timestamp=datetime.now(),
            )
            runs.append(run)

        dto = ExperimentDTO(
            id=exp_id,
            name="algorithm_comparison",
            description="Comparing different anomaly detection algorithms",
            runs=runs,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert len(dto.runs) == 4
        assert dto.runs[0].detector_name == "IsolationForest"
        assert dto.runs[1].detector_name == "OneClassSVM"
        assert dto.runs[2].detector_name == "LocalOutlierFactor"
        assert dto.runs[3].detector_name == "EllipticEnvelope"

    def test_complex_metadata(self):
        """Test complex metadata structures."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        metadata = {
            "project": "credit_card_fraud",
            "version": "2.1.0",
            "author": "data_science_team",
            "environment": {
                "python_version": "3.9.7",
                "packages": {
                    "scikit-learn": "1.0.2",
                    "pandas": "1.4.2",
                    "numpy": "1.21.5",
                },
            },
            "dataset_info": {
                "name": "credit_card_transactions",
                "size": "10GB",
                "samples": 1000000,
                "features": 30,
            },
            "objectives": ["maximize_precision", "minimize_false_positives"],
            "constraints": {
                "max_training_time": "30 minutes",
                "max_memory_usage": "8GB",
            },
        }

        dto = ExperimentDTO(
            id=exp_id,
            name="credit_fraud_detection",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert dto.metadata["project"] == "credit_card_fraud"
        assert dto.metadata["environment"]["python_version"] == "3.9.7"
        assert dto.metadata["dataset_info"]["samples"] == 1000000
        assert len(dto.metadata["objectives"]) == 2

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExperimentDTO(
                name="test_experiment",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExperimentDTO(
                id=str(uuid4()),
                name="test_experiment",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                unknown_field="value",
            )


class TestCreateExperimentDTO:
    """Test suite for CreateExperimentDTO."""

    def test_valid_creation(self):
        """Test creating a valid create experiment DTO."""
        metadata = {"project": "test_project", "version": "1.0"}

        dto = CreateExperimentDTO(
            name="new_experiment",
            description="A new experiment for testing",
            metadata=metadata,
        )

        assert dto.name == "new_experiment"
        assert dto.description == "A new experiment for testing"
        assert dto.metadata == metadata

    def test_default_values(self):
        """Test default values."""
        dto = CreateExperimentDTO(name="basic_experiment")

        assert dto.name == "basic_experiment"
        assert dto.description is None
        assert dto.metadata == {}

    def test_name_validation(self):
        """Test name validation."""
        # Valid name
        dto = CreateExperimentDTO(name="valid_experiment_name")
        assert dto.name == "valid_experiment_name"

        # Invalid: empty name
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="")

        # Invalid: too long name
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="a" * 101)

    def test_description_validation(self):
        """Test description validation."""
        # Valid description
        dto = CreateExperimentDTO(
            name="test_experiment", description="Valid description"
        )
        assert dto.description == "Valid description"

        # Invalid: too long description
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="test_experiment", description="a" * 501)

    def test_optional_description(self):
        """Test optional description field."""
        # With description
        dto_with_desc = CreateExperimentDTO(
            name="test_experiment", description="Test description"
        )
        assert dto_with_desc.description == "Test description"

        # Without description
        dto_without_desc = CreateExperimentDTO(name="test_experiment")
        assert dto_without_desc.description is None

    def test_metadata_handling(self):
        """Test metadata handling."""
        metadata = {
            "project": "anomaly_detection",
            "team": "data_science",
            "priority": "high",
            "tags": ["fraud", "real-time", "production"],
            "config": {"timeout": 3600, "memory_limit": "8GB"},
        }

        dto = CreateExperimentDTO(name="production_experiment", metadata=metadata)

        assert dto.metadata["project"] == "anomaly_detection"
        assert dto.metadata["team"] == "data_science"
        assert dto.metadata["priority"] == "high"
        assert len(dto.metadata["tags"]) == 3
        assert dto.metadata["config"]["timeout"] == 3600

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            CreateExperimentDTO(description="Missing name")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="test_experiment", unknown_field="value")


class TestLeaderboardEntryDTO:
    """Test suite for LeaderboardEntryDTO."""

    def test_valid_creation(self):
        """Test creating a valid leaderboard entry DTO."""
        exp_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.now()
        parameters = {"n_estimators": 100, "contamination": 0.1}

        dto = LeaderboardEntryDTO(
            rank=1,
            experiment_id=exp_id,
            run_id=run_id,
            detector_name="IsolationForest",
            dataset_name="fraud_dataset",
            score=0.95,
            metric_name="f1_score",
            parameters=parameters,
            timestamp=timestamp,
        )

        assert dto.rank == 1
        assert dto.experiment_id == exp_id
        assert dto.run_id == run_id
        assert dto.detector_name == "IsolationForest"
        assert dto.dataset_name == "fraud_dataset"
        assert dto.score == 0.95
        assert dto.metric_name == "f1_score"
        assert dto.parameters == parameters
        assert dto.timestamp == timestamp

    def test_default_parameters(self):
        """Test default parameters value."""
        exp_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.now()

        dto = LeaderboardEntryDTO(
            rank=2,
            experiment_id=exp_id,
            run_id=run_id,
            detector_name="OneClassSVM",
            dataset_name="test_dataset",
            score=0.89,
            metric_name="precision",
            timestamp=timestamp,
        )

        assert dto.parameters == {}

    def test_different_metrics(self):
        """Test different metric types."""
        exp_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.now()

        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
        scores = [0.85, 0.78, 0.92, 0.84, 0.91]

        for i, (metric, score) in enumerate(zip(metrics, scores, strict=False)):
            dto = LeaderboardEntryDTO(
                rank=i + 1,
                experiment_id=exp_id,
                run_id=run_id,
                detector_name="TestDetector",
                dataset_name="test_dataset",
                score=score,
                metric_name=metric,
                timestamp=timestamp,
            )

            assert dto.rank == i + 1
            assert dto.score == score
            assert dto.metric_name == metric

    def test_different_algorithms(self):
        """Test different algorithm entries."""
        exp_id = str(uuid4())
        timestamp = datetime.now()

        algorithms = [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
            "EllipticEnvelope",
        ]

        for i, algorithm in enumerate(algorithms):
            dto = LeaderboardEntryDTO(
                rank=i + 1,
                experiment_id=exp_id,
                run_id=str(uuid4()),
                detector_name=algorithm,
                dataset_name="benchmark_dataset",
                score=0.9 - i * 0.02,
                metric_name="f1_score",
                timestamp=timestamp,
            )

            assert dto.detector_name == algorithm
            assert dto.rank == i + 1
            assert dto.score == 0.9 - i * 0.02

    def test_complex_parameters(self):
        """Test complex parameter structures."""
        exp_id = str(uuid4())
        run_id = str(uuid4())
        timestamp = datetime.now()

        parameters = {
            "algorithm_params": {
                "n_estimators": 200,
                "max_samples": 0.8,
                "contamination": 0.1,
                "random_state": 42,
            },
            "preprocessing": {
                "scaler": "RobustScaler",
                "feature_selection": "SelectKBest",
                "k_features": 15,
            },
            "optimization": {
                "method": "GridSearchCV",
                "cv_folds": 5,
                "scoring": "f1_score",
            },
        }

        dto = LeaderboardEntryDTO(
            rank=1,
            experiment_id=exp_id,
            run_id=run_id,
            detector_name="OptimizedIsolationForest",
            dataset_name="optimized_dataset",
            score=0.96,
            metric_name="f1_score",
            parameters=parameters,
            timestamp=timestamp,
        )

        assert dto.parameters["algorithm_params"]["n_estimators"] == 200
        assert dto.parameters["preprocessing"]["scaler"] == "RobustScaler"
        assert dto.parameters["optimization"]["method"] == "GridSearchCV"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            LeaderboardEntryDTO(
                rank=1,
                experiment_id=str(uuid4()),
                detector_name="IsolationForest",
                dataset_name="test_dataset",
                score=0.85,
                metric_name="accuracy",
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            LeaderboardEntryDTO(
                rank=1,
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                detector_name="IsolationForest",
                dataset_name="test_dataset",
                score=0.85,
                metric_name="accuracy",
                timestamp=datetime.now(),
                unknown_field="value",
            )


class TestExperimentResponseDTO:
    """Test suite for ExperimentResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid experiment response DTO."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()
        metadata = {"project": "test_project", "version": "1.0"}

        dto = ExperimentResponseDTO(
            id=exp_id,
            name="test_experiment",
            description="Test experiment description",
            status="completed",
            total_runs=15,
            best_score=0.95,
            best_metric="f1_score",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert dto.id == exp_id
        assert dto.name == "test_experiment"
        assert dto.description == "Test experiment description"
        assert dto.status == "completed"
        assert dto.total_runs == 15
        assert dto.best_score == 0.95
        assert dto.best_metric == "f1_score"
        assert dto.created_at == created_at
        assert dto.updated_at == updated_at
        assert dto.metadata == metadata

    def test_default_values(self):
        """Test default values."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        dto = ExperimentResponseDTO(
            id=exp_id,
            name="basic_experiment",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert dto.description is None
        assert dto.status == "active"
        assert dto.total_runs == 0
        assert dto.best_score is None
        assert dto.best_metric is None
        assert dto.metadata == {}

    def test_different_statuses(self):
        """Test different experiment statuses."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        statuses = ["active", "completed", "failed", "paused", "archived"]

        for status in statuses:
            dto = ExperimentResponseDTO(
                id=exp_id,
                name="status_test_experiment",
                status=status,
                created_at=created_at,
                updated_at=updated_at,
            )

            assert dto.status == status

    def test_optional_fields(self):
        """Test optional fields."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        # With optional fields
        dto_with_optional = ExperimentResponseDTO(
            id=exp_id,
            name="experiment_with_optional",
            description="Has optional fields",
            best_score=0.88,
            best_metric="precision",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert dto_with_optional.description == "Has optional fields"
        assert dto_with_optional.best_score == 0.88
        assert dto_with_optional.best_metric == "precision"

        # Without optional fields
        dto_without_optional = ExperimentResponseDTO(
            id=exp_id,
            name="experiment_without_optional",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert dto_without_optional.description is None
        assert dto_without_optional.best_score is None
        assert dto_without_optional.best_metric is None

    def test_experiment_progress(self):
        """Test experiment progress tracking."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        # Early experiment
        early_dto = ExperimentResponseDTO(
            id=exp_id,
            name="early_experiment",
            status="active",
            total_runs=3,
            best_score=0.72,
            best_metric="accuracy",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert early_dto.total_runs == 3
        assert early_dto.best_score == 0.72
        assert early_dto.status == "active"

        # Mature experiment
        mature_dto = ExperimentResponseDTO(
            id=exp_id,
            name="mature_experiment",
            status="completed",
            total_runs=50,
            best_score=0.96,
            best_metric="f1_score",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert mature_dto.total_runs == 50
        assert mature_dto.best_score == 0.96
        assert mature_dto.status == "completed"

    def test_complex_metadata(self):
        """Test complex metadata structures."""
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        metadata = {
            "summary": {
                "algorithms_tested": 8,
                "datasets_used": 3,
                "total_training_time": "2 hours 45 minutes",
                "best_algorithm": "OptimizedIsolationForest",
            },
            "performance": {
                "baseline_score": 0.73,
                "improvement": 0.23,
                "stability": 0.85,
            },
            "resources": {
                "cpu_hours": 12.5,
                "memory_peak": "16GB",
                "storage_used": "2.3GB",
            },
        }

        dto = ExperimentResponseDTO(
            id=exp_id,
            name="complex_experiment",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        assert dto.metadata["summary"]["algorithms_tested"] == 8
        assert dto.metadata["performance"]["improvement"] == 0.23
        assert dto.metadata["resources"]["cpu_hours"] == 12.5

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExperimentResponseDTO(
                name="test_experiment",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExperimentResponseDTO(
                id=str(uuid4()),
                name="test_experiment",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                unknown_field="value",
            )


class TestExperimentDTOIntegration:
    """Test integration scenarios for experiment DTOs."""

    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        # Create experiment
        create_dto = CreateExperimentDTO(
            name="fraud_detection_experiment",
            description="Comprehensive fraud detection experiment",
            metadata={"project": "fraud_detection", "priority": "high"},
        )

        # Simulate created experiment
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = created_at

        experiment_dto = ExperimentDTO(
            id=exp_id,
            name=create_dto.name,
            description=create_dto.description,
            created_at=created_at,
            updated_at=updated_at,
            metadata=create_dto.metadata,
        )

        # Add runs to experiment
        runs = []
        algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]

        for algorithm in algorithms:
            run = RunDTO(
                id=str(uuid4()),
                detector_name=algorithm,
                dataset_name="fraud_dataset",
                parameters={"random_state": 42},
                metrics={"f1_score": 0.8 + len(runs) * 0.02},
                timestamp=datetime.now(),
            )
            runs.append(run)

        experiment_dto.runs = runs

        # Create response DTO
        response_dto = ExperimentResponseDTO(
            id=experiment_dto.id,
            name=experiment_dto.name,
            description=experiment_dto.description,
            status="completed",
            total_runs=len(experiment_dto.runs),
            best_score=max(run.metrics["f1_score"] for run in experiment_dto.runs),
            best_metric="f1_score",
            created_at=experiment_dto.created_at,
            updated_at=experiment_dto.updated_at,
            metadata=experiment_dto.metadata,
        )

        # Verify lifecycle consistency
        assert response_dto.name == create_dto.name
        assert response_dto.description == create_dto.description
        assert response_dto.total_runs == len(algorithms)
        assert response_dto.best_score == 0.84  # 0.8 + 2 * 0.02
        assert response_dto.metadata == create_dto.metadata

    def test_leaderboard_generation(self):
        """Test leaderboard generation from experiment."""
        exp_id = str(uuid4())

        # Create experiment with runs
        runs = []
        algorithms = [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
            "EllipticEnvelope",
        ]
        scores = [0.95, 0.88, 0.92, 0.85]

        for algorithm, score in zip(algorithms, scores, strict=False):
            run = RunDTO(
                id=str(uuid4()),
                detector_name=algorithm,
                dataset_name="benchmark_dataset",
                parameters={"contamination": 0.1},
                metrics={"f1_score": score},
                timestamp=datetime.now(),
            )
            runs.append(run)

        experiment = ExperimentDTO(
            id=exp_id,
            name="benchmark_experiment",
            runs=runs,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Generate leaderboard entries
        leaderboard_entries = []
        sorted_runs = sorted(
            experiment.runs, key=lambda r: r.metrics["f1_score"], reverse=True
        )

        for rank, run in enumerate(sorted_runs, 1):
            entry = LeaderboardEntryDTO(
                rank=rank,
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

        # Verify leaderboard ordering
        assert leaderboard_entries[0].rank == 1
        assert leaderboard_entries[0].detector_name == "IsolationForest"
        assert leaderboard_entries[0].score == 0.95

        assert leaderboard_entries[1].rank == 2
        assert leaderboard_entries[1].detector_name == "LocalOutlierFactor"
        assert leaderboard_entries[1].score == 0.92

        assert leaderboard_entries[2].rank == 3
        assert leaderboard_entries[2].detector_name == "OneClassSVM"
        assert leaderboard_entries[2].score == 0.88

        assert leaderboard_entries[3].rank == 4
        assert leaderboard_entries[3].detector_name == "EllipticEnvelope"
        assert leaderboard_entries[3].score == 0.85

    def test_experiment_tracking_workflow(self):
        """Test experiment tracking workflow."""
        # Create experiment
        create_dto = CreateExperimentDTO(
            name="hyperparameter_tuning_experiment",
            description="Systematic hyperparameter tuning",
            metadata={"optimization_method": "grid_search"},
        )

        exp_id = str(uuid4())
        experiment = ExperimentDTO(
            id=exp_id,
            name=create_dto.name,
            description=create_dto.description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=create_dto.metadata,
        )

        # Simulate hyperparameter tuning runs
        n_estimators_values = [50, 100, 200]
        contamination_values = [0.05, 0.1, 0.15]

        runs = []
        for n_est in n_estimators_values:
            for cont in contamination_values:
                run = RunDTO(
                    id=str(uuid4()),
                    detector_name="IsolationForest",
                    dataset_name="tuning_dataset",
                    parameters={"n_estimators": n_est, "contamination": cont},
                    metrics={
                        "f1_score": 0.7 + (n_est / 200) * 0.2 + (0.1 / cont) * 0.05
                    },
                    timestamp=datetime.now(),
                )
                runs.append(run)

        experiment.runs = runs

        # Find best run
        best_run = max(experiment.runs, key=lambda r: r.metrics["f1_score"])

        # Create response with best results
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
            metadata={
                **experiment.metadata,
                "best_parameters": best_run.parameters,
                "grid_search_size": len(runs),
            },
        )

        # Verify tracking workflow
        assert response.total_runs == 9  # 3 * 3 combinations
        assert response.best_score == best_run.metrics["f1_score"]
        assert response.metadata["best_parameters"] == best_run.parameters
        assert response.metadata["grid_search_size"] == 9

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create experiment DTO
        exp_id = str(uuid4())
        created_at = datetime.now()
        updated_at = datetime.now()

        run = RunDTO(
            id=str(uuid4()),
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"n_estimators": 100},
            metrics={"f1_score": 0.85},
            timestamp=datetime.now(),
        )

        original_dto = ExperimentDTO(
            id=exp_id,
            name="serialization_test",
            description="Test serialization",
            runs=[run],
            created_at=created_at,
            updated_at=updated_at,
            metadata={"test": "value"},
        )

        # Serialize to dict
        dto_dict = original_dto.model_dump()

        assert dto_dict["id"] == exp_id
        assert dto_dict["name"] == "serialization_test"
        assert dto_dict["description"] == "Test serialization"
        assert len(dto_dict["runs"]) == 1
        assert dto_dict["metadata"] == {"test": "value"}

        # Deserialize from dict
        restored_dto = ExperimentDTO.model_validate(dto_dict)

        assert restored_dto.id == original_dto.id
        assert restored_dto.name == original_dto.name
        assert restored_dto.description == original_dto.description
        assert len(restored_dto.runs) == len(original_dto.runs)
        assert restored_dto.metadata == original_dto.metadata

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum name length
        create_dto = CreateExperimentDTO(name="a")
        assert create_dto.name == "a"

        # Maximum name length
        create_dto = CreateExperimentDTO(name="a" * 100)
        assert len(create_dto.name) == 100

        # Maximum description length
        create_dto = CreateExperimentDTO(name="test", description="a" * 500)
        assert len(create_dto.description) == 500

        # Zero runs
        experiment = ExperimentDTO(
            id=str(uuid4()),
            name="empty_experiment",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert len(experiment.runs) == 0

        # Response with no best results
        response = ExperimentResponseDTO(
            id=str(uuid4()),
            name="no_results_experiment",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_runs=0,
        )
        assert response.best_score is None
        assert response.best_metric is None
