"""Comprehensive unit tests for Experiment domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from mlops.domain.entities.experiment import Experiment, ExperimentStatus


@pytest.fixture
def sample_experiment():
    """Sample experiment instance for testing."""
    return Experiment(
        name="Test Experiment",
        description="A test experiment for unit testing",
        created_by="test_user",
        team="ML Team"
    )


@pytest.fixture
def experiment_with_data():
    """Experiment with parameters, metrics, and artifacts."""
    experiment = Experiment(
        name="Data Experiment",
        description="Experiment with test data",
        created_by="data_scientist"
    )
    experiment.parameters = {"learning_rate": 0.01, "batch_size": 32}
    experiment.metrics = {"accuracy": 0.85, "loss": 0.15}
    experiment.artifacts = {"model_path": "/models/test_model.pkl"}
    experiment.tags = ["test", "classification"]
    return experiment


class TestExperimentStatus:
    """Test cases for ExperimentStatus enum."""

    def test_experiment_status_values(self):
        """Test all ExperimentStatus enum values."""
        assert ExperimentStatus.CREATED.value == "created"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.FAILED.value == "failed"
        assert ExperimentStatus.CANCELLED.value == "cancelled"


class TestExperiment:
    """Test cases for Experiment domain entity."""

    def test_initialization_defaults(self):
        """Test experiment initialization with default values."""
        experiment = Experiment()
        
        assert isinstance(experiment.id, UUID)
        assert experiment.name.startswith("experiment_")
        assert experiment.description == ""
        assert experiment.status == ExperimentStatus.CREATED
        assert experiment.parameters == {}
        assert experiment.metrics == {}
        assert experiment.artifacts == {}
        assert experiment.tags == []
        assert experiment.model_id is None
        assert isinstance(experiment.created_at, datetime)
        assert isinstance(experiment.updated_at, datetime)
        assert experiment.started_at is None
        assert experiment.completed_at is None
        assert experiment.created_by == ""
        assert experiment.team == ""

    def test_initialization_with_data(self):
        """Test experiment initialization with provided data."""
        experiment_id = uuid4()
        model_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        updated_at = datetime(2024, 1, 15, 11, 0, 0)
        
        experiment = Experiment(
            id=experiment_id,
            name="Custom Experiment",
            description="Custom description",
            status=ExperimentStatus.RUNNING,
            parameters={"param1": "value1"},
            metrics={"metric1": 0.9},
            artifacts={"artifact1": "path1"},
            tags=["tag1", "tag2"],
            model_id=model_id,
            created_at=created_at,
            updated_at=updated_at,
            created_by="scientist",
            team="research"
        )
        
        assert experiment.id == experiment_id
        assert experiment.name == "Custom Experiment"
        assert experiment.description == "Custom description"
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.parameters == {"param1": "value1"}
        assert experiment.metrics == {"metric1": 0.9}
        assert experiment.artifacts == {"artifact1": "path1"}
        assert experiment.tags == ["tag1", "tag2"]
        assert experiment.model_id == model_id
        assert experiment.created_at == created_at
        assert experiment.updated_at == updated_at
        assert experiment.created_by == "scientist"
        assert experiment.team == "research"

    def test_post_init_auto_name_generation(self):
        """Test automatic name generation when name is empty."""
        experiment = Experiment(name="")
        
        assert experiment.name.startswith("experiment_")
        assert len(experiment.name) > 11  # "experiment_" + hex digits

    def test_post_init_validation_empty_name_after_strip(self):
        """Test validation fails for name that's empty after stripping."""
        with pytest.raises(ValueError, match="Experiment name cannot be empty"):
            Experiment(name="   ")

    def test_post_init_validation_name_too_long(self):
        """Test validation fails for name that's too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Experiment name cannot exceed 100 characters"):
            Experiment(name=long_name)

    def test_post_init_validation_invalid_status(self):
        """Test validation fails for invalid status."""
        with pytest.raises(ValueError, match="Invalid experiment status"):
            # This would normally fail at the dataclass level, but let's test the validation
            experiment = Experiment(name="Test")
            experiment.status = "invalid_status"
            experiment._validate_experiment()

    def test_start_success(self, sample_experiment):
        """Test starting experiment successfully."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.start()
            
            assert sample_experiment.status == ExperimentStatus.RUNNING
            assert sample_experiment.started_at == mock_now
            assert sample_experiment.updated_at == mock_now

    def test_start_from_non_created_status(self, sample_experiment):
        """Test starting experiment from non-CREATED status fails."""
        sample_experiment.status = ExperimentStatus.RUNNING
        
        with pytest.raises(ValueError, match="Cannot start experiment in running status"):
            sample_experiment.start()

    def test_complete_success(self, sample_experiment):
        """Test completing experiment successfully."""
        # First start the experiment
        sample_experiment.start()
        
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 13, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            final_metrics = {"final_accuracy": 0.92, "final_loss": 0.08}
            sample_experiment.complete(final_metrics)
            
            assert sample_experiment.status == ExperimentStatus.COMPLETED
            assert sample_experiment.completed_at == mock_now
            assert sample_experiment.updated_at == mock_now
            assert sample_experiment.metrics["final_accuracy"] == 0.92
            assert sample_experiment.metrics["final_loss"] == 0.08

    def test_complete_without_metrics(self, sample_experiment):
        """Test completing experiment without final metrics."""
        sample_experiment.start()
        sample_experiment.metrics = {"initial_accuracy": 0.8}
        
        sample_experiment.complete()
        
        assert sample_experiment.status == ExperimentStatus.COMPLETED
        assert sample_experiment.metrics == {"initial_accuracy": 0.8}

    def test_complete_from_non_running_status(self, sample_experiment):
        """Test completing experiment from non-RUNNING status fails."""
        with pytest.raises(ValueError, match="Cannot complete experiment in created status"):
            sample_experiment.complete()

    def test_fail_experiment(self, sample_experiment):
        """Test failing an experiment."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            error_message = "Model training failed"
            sample_experiment.fail(error_message)
            
            assert sample_experiment.status == ExperimentStatus.FAILED
            assert sample_experiment.completed_at == mock_now
            assert sample_experiment.updated_at == mock_now
            assert sample_experiment.artifacts["error_message"] == error_message

    def test_fail_experiment_without_error_message(self, sample_experiment):
        """Test failing experiment without error message."""
        sample_experiment.fail()
        
        assert sample_experiment.status == ExperimentStatus.FAILED
        assert "error_message" not in sample_experiment.artifacts

    def test_cancel_success(self, sample_experiment):
        """Test cancelling experiment successfully."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.cancel()
            
            assert sample_experiment.status == ExperimentStatus.CANCELLED
            assert sample_experiment.completed_at == mock_now
            assert sample_experiment.updated_at == mock_now

    def test_cancel_running_experiment(self, sample_experiment):
        """Test cancelling a running experiment."""
        sample_experiment.start()
        
        sample_experiment.cancel()
        
        assert sample_experiment.status == ExperimentStatus.CANCELLED

    def test_cancel_completed_experiment(self, sample_experiment):
        """Test cancelling completed experiment fails."""
        sample_experiment.start()
        sample_experiment.complete()
        
        with pytest.raises(ValueError, match="Cannot cancel experiment in completed status"):
            sample_experiment.cancel()

    def test_cancel_failed_experiment(self, sample_experiment):
        """Test cancelling failed experiment fails."""
        sample_experiment.fail()
        
        with pytest.raises(ValueError, match="Cannot cancel experiment in failed status"):
            sample_experiment.cancel()

    def test_add_metric_success(self, sample_experiment):
        """Test adding metrics successfully."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.add_metric("accuracy", 0.85)
            sample_experiment.add_metric("precision", 0.82)
            sample_experiment.add_metric("recall", 0.88)
            
            assert sample_experiment.metrics["accuracy"] == 0.85
            assert sample_experiment.metrics["precision"] == 0.82
            assert sample_experiment.metrics["recall"] == 0.88
            assert sample_experiment.updated_at == mock_now

    def test_add_metric_with_integer(self, sample_experiment):
        """Test adding metric with integer value."""
        sample_experiment.add_metric("epochs", 100)
        
        assert sample_experiment.metrics["epochs"] == 100.0

    def test_add_metric_invalid_name(self, sample_experiment):
        """Test adding metric with invalid name fails."""
        with pytest.raises(ValueError, match="Metric name must be a non-empty string"):
            sample_experiment.add_metric("", 0.85)
        
        with pytest.raises(ValueError, match="Metric name must be a non-empty string"):
            sample_experiment.add_metric(None, 0.85)

    def test_add_metric_invalid_value(self, sample_experiment):
        """Test adding metric with invalid value fails."""
        with pytest.raises(ValueError, match="Metric value must be a number"):
            sample_experiment.add_metric("accuracy", "invalid")
        
        with pytest.raises(ValueError, match="Metric value must be a number"):
            sample_experiment.add_metric("accuracy", None)

    def test_add_artifact_success(self, sample_experiment):
        """Test adding artifacts successfully."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.add_artifact("model", "/path/to/model.pkl")
            sample_experiment.add_artifact("plot", "s3://bucket/plot.png")
            
            assert sample_experiment.artifacts["model"] == "/path/to/model.pkl"
            assert sample_experiment.artifacts["plot"] == "s3://bucket/plot.png"
            assert sample_experiment.updated_at == mock_now

    def test_add_artifact_invalid_name(self, sample_experiment):
        """Test adding artifact with invalid name fails."""
        with pytest.raises(ValueError, match="Artifact name must be a non-empty string"):
            sample_experiment.add_artifact("", "/path/to/artifact")
        
        with pytest.raises(ValueError, match="Artifact name must be a non-empty string"):
            sample_experiment.add_artifact(None, "/path/to/artifact")

    def test_add_artifact_invalid_path(self, sample_experiment):
        """Test adding artifact with invalid path fails."""
        with pytest.raises(ValueError, match="Artifact path must be a non-empty string"):
            sample_experiment.add_artifact("model", "")
        
        with pytest.raises(ValueError, match="Artifact path must be a non-empty string"):
            sample_experiment.add_artifact("model", None)

    def test_add_tag_success(self, sample_experiment):
        """Test adding tags successfully."""
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.add_tag("classification")
            sample_experiment.add_tag("production")
            
            assert "classification" in sample_experiment.tags
            assert "production" in sample_experiment.tags
            assert len(sample_experiment.tags) == 2
            assert sample_experiment.updated_at == mock_now

    def test_add_tag_duplicate(self, sample_experiment):
        """Test adding duplicate tag doesn't create duplicates."""
        sample_experiment.add_tag("test")
        sample_experiment.add_tag("test")  # Duplicate
        
        assert sample_experiment.tags.count("test") == 1

    def test_add_tag_invalid(self, sample_experiment):
        """Test adding invalid tag fails."""
        with pytest.raises(ValueError, match="Tag must be a non-empty string"):
            sample_experiment.add_tag("")
        
        with pytest.raises(ValueError, match="Tag must be a non-empty string"):
            sample_experiment.add_tag(None)

    def test_remove_tag_success(self, sample_experiment):
        """Test removing tags successfully."""
        sample_experiment.tags = ["tag1", "tag2", "tag3"]
        
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 19, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_experiment.remove_tag("tag2")
            
            assert "tag2" not in sample_experiment.tags
            assert len(sample_experiment.tags) == 2
            assert sample_experiment.updated_at == mock_now

    def test_remove_tag_nonexistent(self, sample_experiment):
        """Test removing non-existent tag does nothing."""
        sample_experiment.tags = ["tag1", "tag2"]
        original_updated_at = sample_experiment.updated_at
        
        sample_experiment.remove_tag("nonexistent")
        
        assert len(sample_experiment.tags) == 2
        assert sample_experiment.updated_at == original_updated_at

    def test_duration_property_not_started(self, sample_experiment):
        """Test duration property when experiment hasn't started."""
        assert sample_experiment.duration is None

    def test_duration_property_running(self, sample_experiment):
        """Test duration property for running experiment."""
        start_time = datetime(2024, 1, 15, 10, 0, 0)
        
        with patch('mlops.domain.entities.experiment.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = start_time
            sample_experiment.start()
            
            # Mock current time as 30 minutes later
            current_time = start_time + timedelta(minutes=30)
            mock_datetime.utcnow.return_value = current_time
            
            duration = sample_experiment.duration
            assert duration == 1800.0  # 30 minutes in seconds

    def test_duration_property_completed(self, sample_experiment):
        """Test duration property for completed experiment."""
        start_time = datetime(2024, 1, 15, 10, 0, 0)
        complete_time = datetime(2024, 1, 15, 11, 0, 0)
        
        sample_experiment.started_at = start_time
        sample_experiment.completed_at = complete_time
        
        duration = sample_experiment.duration
        assert duration == 3600.0  # 1 hour in seconds

    def test_is_active_property(self, sample_experiment):
        """Test is_active property."""
        assert sample_experiment.is_active is False
        
        sample_experiment.start()
        assert sample_experiment.is_active is True
        
        sample_experiment.complete()
        assert sample_experiment.is_active is False

    def test_is_finished_property(self, sample_experiment):
        """Test is_finished property."""
        assert sample_experiment.is_finished is False
        
        sample_experiment.start()
        assert sample_experiment.is_finished is False
        
        sample_experiment.complete()
        assert sample_experiment.is_finished is True
        
        # Test other finished states
        sample_experiment2 = Experiment(name="Test 2")
        sample_experiment2.fail()
        assert sample_experiment2.is_finished is True
        
        sample_experiment3 = Experiment(name="Test 3")
        sample_experiment3.cancel()
        assert sample_experiment3.is_finished is True

    def test_get_best_metric_existing(self, sample_experiment):
        """Test getting existing metric."""
        sample_experiment.metrics = {"accuracy": 0.92, "loss": 0.08}
        
        assert sample_experiment.get_best_metric("accuracy") == 0.92
        assert sample_experiment.get_best_metric("loss") == 0.08

    def test_get_best_metric_nonexistent(self, sample_experiment):
        """Test getting non-existent metric returns None."""
        assert sample_experiment.get_best_metric("nonexistent") is None

    def test_to_dict(self, experiment_with_data):
        """Test converting experiment to dictionary."""
        experiment_with_data.start()
        experiment_with_data.complete()
        
        result = experiment_with_data.to_dict()
        
        assert result["id"] == str(experiment_with_data.id)
        assert result["name"] == "Data Experiment"
        assert result["description"] == "Experiment with test data"
        assert result["status"] == "completed"
        assert result["parameters"] == {"learning_rate": 0.01, "batch_size": 32}
        assert result["metrics"] == {"accuracy": 0.85, "loss": 0.15}
        assert result["artifacts"] == {"model_path": "/models/test_model.pkl"}
        assert result["tags"] == ["test", "classification"]
        assert result["model_id"] is None
        assert result["created_by"] == "data_scientist"
        assert result["team"] == ""
        assert isinstance(result["duration"], float)
        assert result["is_active"] is False
        assert result["is_finished"] is True

    def test_to_dict_with_model_id(self, sample_experiment):
        """Test to_dict with model_id set."""
        model_id = uuid4()
        sample_experiment.model_id = model_id
        
        result = sample_experiment.to_dict()
        
        assert result["model_id"] == str(model_id)

    def test_from_dict_minimal(self):
        """Test creating experiment from minimal dictionary."""
        data = {
            "name": "From Dict Experiment",
            "description": "Created from dictionary"
        }
        
        experiment = Experiment.from_dict(data)
        
        assert experiment.name == "From Dict Experiment"
        assert experiment.description == "Created from dictionary"
        assert experiment.status == ExperimentStatus.CREATED
        assert isinstance(experiment.id, UUID)

    def test_from_dict_complete(self):
        """Test creating experiment from complete dictionary."""
        experiment_id = uuid4()
        model_id = uuid4()
        
        data = {
            "id": str(experiment_id),
            "name": "Complete Experiment",
            "description": "Complete description",
            "status": "running",
            "parameters": {"param1": "value1"},
            "metrics": {"accuracy": 0.9},
            "artifacts": {"model": "/path/model.pkl"},
            "tags": ["tag1", "tag2"],
            "model_id": str(model_id),
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T11:00:00",
            "started_at": "2024-01-15T10:30:00",
            "completed_at": "2024-01-15T12:00:00",
            "created_by": "scientist",
            "team": "research"
        }
        
        experiment = Experiment.from_dict(data)
        
        assert experiment.id == experiment_id
        assert experiment.name == "Complete Experiment"
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.model_id == model_id
        assert experiment.created_at == datetime(2024, 1, 15, 10, 0, 0)
        assert experiment.updated_at == datetime(2024, 1, 15, 11, 0, 0)
        assert experiment.started_at == datetime(2024, 1, 15, 10, 30, 0)
        assert experiment.completed_at == datetime(2024, 1, 15, 12, 0, 0)

    def test_from_dict_with_none_values(self):
        """Test from_dict handles None values correctly."""
        data = {
            "name": "Test Experiment",
            "model_id": None,
            "started_at": None,
            "completed_at": None
        }
        
        experiment = Experiment.from_dict(data)
        
        assert experiment.model_id is None
        assert experiment.started_at is None
        assert experiment.completed_at is None

    def test_str_representation(self, sample_experiment):
        """Test string representation."""
        str_repr = str(sample_experiment)
        
        assert "Experiment(" in str_repr
        assert sample_experiment.id.hex[:8] in str_repr
        assert "Test Experiment" in str_repr
        assert "created" in str_repr

    def test_repr_representation(self, experiment_with_data):
        """Test detailed representation."""
        repr_str = repr(experiment_with_data)
        
        assert f"id={experiment_with_data.id}" in repr_str
        assert "name='Data Experiment'" in repr_str
        assert "status=created" in repr_str
        assert "metrics_count=2" in repr_str
        assert "artifacts_count=1" in repr_str

    def test_experiment_lifecycle_integration(self, sample_experiment):
        """Test complete experiment lifecycle."""
        # Add initial data
        sample_experiment.add_metric("initial_accuracy", 0.7)
        sample_experiment.add_artifact("dataset", "/data/train.csv")
        sample_experiment.add_tag("experiment")
        
        # Start experiment
        sample_experiment.start()
        assert sample_experiment.is_active
        assert not sample_experiment.is_finished
        
        # Add more metrics during training
        sample_experiment.add_metric("epoch_1_loss", 0.5)
        sample_experiment.add_metric("epoch_2_loss", 0.3)
        
        # Complete with final metrics
        final_metrics = {"final_accuracy": 0.92, "final_loss": 0.08}
        sample_experiment.complete(final_metrics)
        
        assert not sample_experiment.is_active
        assert sample_experiment.is_finished
        assert sample_experiment.status == ExperimentStatus.COMPLETED
        assert sample_experiment.metrics["final_accuracy"] == 0.92
        assert sample_experiment.duration is not None