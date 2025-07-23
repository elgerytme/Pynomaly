"""Comprehensive unit tests for TrainingJob entity."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from machine_learning.domain.entities.training_job import (
    TrainingJob,
    TrainingStatus,
    TrainingPriority,
    ResourceUsage,
    TrainingMetrics,
    AlgorithmResult
)


class TestResourceUsage:
    """Test cases for ResourceUsage dataclass."""

    def test_resource_usage_defaults(self):
        """Test ResourceUsage with default values."""
        usage = ResourceUsage()
        
        assert usage.cpu_time == 0.0
        assert usage.memory_peak == 0.0
        assert usage.gpu_memory_peak == 0.0
        assert usage.disk_io == 0.0
        assert usage.network_io == 0.0

    def test_resource_usage_with_values(self):
        """Test ResourceUsage with custom values."""
        usage = ResourceUsage(
            cpu_time=120.5,
            memory_peak=512.0,
            gpu_memory_peak=2048.0,
            disk_io=100.5,
            network_io=50.2
        )
        
        assert usage.cpu_time == 120.5
        assert usage.memory_peak == 512.0
        assert usage.gpu_memory_peak == 2048.0
        assert usage.disk_io == 100.5
        assert usage.network_io == 50.2

    def test_resource_usage_to_dict(self):
        """Test ResourceUsage to_dict conversion."""
        usage = ResourceUsage(
            cpu_time=120.5,
            memory_peak=512.0,
            gpu_memory_peak=2048.0,
            disk_io=100.5,
            network_io=50.2
        )
        
        expected = {
            "cpu_time": 120.5,
            "memory_peak": 512.0,
            "gpu_memory_peak": 2048.0,
            "disk_io": 100.5,
            "network_io": 50.2
        }
        
        assert usage.to_dict() == expected


class TestTrainingMetrics:
    """Test cases for TrainingMetrics dataclass."""

    def test_training_metrics_defaults(self):
        """Test TrainingMetrics with default values."""
        metrics = TrainingMetrics()
        
        assert metrics.best_score is None
        assert metrics.final_score is None
        assert metrics.validation_scores == []
        assert metrics.training_loss == []
        assert metrics.validation_loss == []
        assert metrics.epochs_completed == 0
        assert metrics.early_stopping_epoch is None
        assert metrics.convergence_metrics == {}

    def test_training_metrics_with_values(self):
        """Test TrainingMetrics with custom values."""
        metrics = TrainingMetrics(
            best_score=0.95,
            final_score=0.92,
            validation_scores=[0.85, 0.90, 0.92],
            training_loss=[0.5, 0.3, 0.2],
            validation_loss=[0.6, 0.4, 0.3],
            epochs_completed=50,
            early_stopping_epoch=45,
            convergence_metrics={"tolerance": 0.001}
        )
        
        assert metrics.best_score == 0.95
        assert metrics.final_score == 0.92
        assert metrics.validation_scores == [0.85, 0.90, 0.92]
        assert metrics.training_loss == [0.5, 0.3, 0.2]
        assert metrics.validation_loss == [0.6, 0.4, 0.3]
        assert metrics.epochs_completed == 50
        assert metrics.early_stopping_epoch == 45
        assert metrics.convergence_metrics == {"tolerance": 0.001}

    def test_training_metrics_to_dict(self):
        """Test TrainingMetrics to_dict conversion."""
        metrics = TrainingMetrics(
            best_score=0.95,
            final_score=0.92,
            validation_scores=[0.85, 0.90, 0.92],
            training_loss=[0.5, 0.3, 0.2],
            validation_loss=[0.6, 0.4, 0.3],
            epochs_completed=50,
            early_stopping_epoch=45,
            convergence_metrics={"tolerance": 0.001}
        )
        
        expected = {
            "best_score": 0.95,
            "final_score": 0.92,
            "validation_scores": [0.85, 0.90, 0.92],
            "training_loss": [0.5, 0.3, 0.2],
            "validation_loss": [0.6, 0.4, 0.3],
            "epochs_completed": 50,
            "early_stopping_epoch": 45,
            "convergence_metrics": {"tolerance": 0.001}
        }
        
        assert metrics.to_dict() == expected


class TestAlgorithmResult:
    """Test cases for AlgorithmResult dataclass."""

    def test_algorithm_result_minimal(self):
        """Test AlgorithmResult with minimal required fields."""
        result = AlgorithmResult(algorithm_name="isolation_forest")
        
        assert result.algorithm_name == "isolation_forest"
        assert result.model_id is None
        assert result.hyperparameters is None
        assert result.metrics is None
        assert result.training_time == 0.0
        assert result.optimization_history == []
        assert result.status == TrainingStatus.PENDING
        assert result.error_message is None

    def test_algorithm_result_complete(self):
        """Test AlgorithmResult with all fields."""
        metrics = TrainingMetrics(best_score=0.95)
        hyperparameters = Mock()
        hyperparameters.to_dict.return_value = {"param1": "value1"}
        
        result = AlgorithmResult(
            algorithm_name="isolation_forest",
            model_id="model_123",
            hyperparameters=hyperparameters,
            metrics=metrics,
            training_time=45.5,
            optimization_history=[{"trial": 1, "score": 0.9}],
            status=TrainingStatus.COMPLETED,
            error_message=None
        )
        
        assert result.algorithm_name == "isolation_forest"
        assert result.model_id == "model_123"
        assert result.hyperparameters == hyperparameters
        assert result.metrics == metrics
        assert result.training_time == 45.5
        assert result.optimization_history == [{"trial": 1, "score": 0.9}]
        assert result.status == TrainingStatus.COMPLETED

    def test_algorithm_result_to_dict(self):
        """Test AlgorithmResult to_dict conversion."""
        metrics = TrainingMetrics(best_score=0.95)
        hyperparameters = Mock()
        hyperparameters.to_dict.return_value = {"param1": "value1"}
        
        result = AlgorithmResult(
            algorithm_name="isolation_forest",
            model_id="model_123",
            hyperparameters=hyperparameters,
            metrics=metrics,
            training_time=45.5,
            optimization_history=[{"trial": 1, "score": 0.9}],
            status=TrainingStatus.COMPLETED,
            error_message="test error"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["algorithm_name"] == "isolation_forest"
        assert result_dict["model_id"] == "model_123"
        assert result_dict["hyperparameters"] == {"param1": "value1"}
        assert result_dict["metrics"]["best_score"] == 0.95
        assert result_dict["training_time"] == 45.5
        assert result_dict["optimization_history"] == [{"trial": 1, "score": 0.9}]
        assert result_dict["status"] == "completed"
        assert result_dict["error_message"] == "test error"

    def test_algorithm_result_to_dict_with_none_values(self):
        """Test AlgorithmResult to_dict with None values."""
        result = AlgorithmResult(algorithm_name="isolation_forest")
        result_dict = result.to_dict()
        
        assert result_dict["hyperparameters"] is None
        assert result_dict["metrics"] is None


class TestTrainingJob:
    """Test cases for TrainingJob entity."""

    def test_training_job_minimal_creation(self):
        """Test TrainingJob creation with minimal required fields."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert job.id == "job_123"
        assert job.dataset_id == "dataset_456"
        assert job.name is None
        assert job.description is None
        assert job.algorithms == []
        assert job.config is None
        assert job.priority == TrainingPriority.NORMAL
        assert job.status == TrainingStatus.PENDING
        assert job.progress == 0.0
        assert job.current_algorithm is None
        assert job.current_trial is None
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
        assert job.updated_at is not None
        assert job.estimated_completion is None
        assert job.algorithm_results == []
        assert job.best_model_id is None
        assert job.best_algorithm is None
        assert job.final_metrics is None
        assert isinstance(job.resource_usage, ResourceUsage)
        assert job.worker_id is None
        assert job.gpu_ids == []
        assert job.error_message is None
        assert job.error_details == {}
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.tags == []
        assert job.user_id is None
        assert job.experiment_id is None
        assert job.parent_job_id is None
        assert job.child_job_ids == []

    def test_training_job_complete_creation(self):
        """Test TrainingJob creation with all fields."""
        created_at = datetime.utcnow()
        config = Mock()
        
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            name="Test Job",
            description="A test training job",
            algorithms=["isolation_forest", "local_outlier_factor"],
            config=config,
            priority=TrainingPriority.HIGH,
            status=TrainingStatus.RUNNING,
            progress=50.0,
            current_algorithm="isolation_forest",
            current_trial=5,
            created_at=created_at,
            worker_id="worker_1",
            gpu_ids=[0, 1],
            tags=["experiment", "production"],
            user_id="user_123",
            experiment_id="exp_456",
            parent_job_id="parent_789",
            child_job_ids=["child_1", "child_2"],
            max_retries=5
        )
        
        assert job.name == "Test Job"
        assert job.description == "A test training job"
        assert job.algorithms == ["isolation_forest", "local_outlier_factor"]
        assert job.config == config
        assert job.priority == TrainingPriority.HIGH
        assert job.status == TrainingStatus.RUNNING
        assert job.progress == 50.0
        assert job.current_algorithm == "isolation_forest"
        assert job.current_trial == 5
        assert job.created_at == created_at
        assert job.worker_id == "worker_1"
        assert job.gpu_ids == [0, 1]
        assert job.tags == ["experiment", "production"]
        assert job.user_id == "user_123"
        assert job.experiment_id == "exp_456"
        assert job.parent_job_id == "parent_789"
        assert job.child_job_ids == ["child_1", "child_2"]
        assert job.max_retries == 5

    def test_training_job_post_init_updated_at(self):
        """Test that updated_at is set to created_at if not provided."""
        created_at = datetime.utcnow()
        job = TrainingJob(id="job_123", dataset_id="dataset_456", created_at=created_at)
        
        assert job.updated_at == created_at

    def test_training_job_post_init_legacy_results_conversion(self):
        """Test conversion of legacy results to algorithm_results."""
        legacy_results = [
            {
                "algorithm": "isolation_forest",
                "model_id": "model_1",
                "training_time": 30.0,
                "optimization_history": [{"trial": 1}]
            },
            {
                "algorithm": "local_outlier_factor",
                "model_id": "model_2",
                "training_time": 25.0
            }
        ]
        
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            results=legacy_results
        )
        
        assert len(job.algorithm_results) == 2
        assert job.algorithm_results[0].algorithm_name == "isolation_forest"
        assert job.algorithm_results[0].model_id == "model_1"
        assert job.algorithm_results[0].training_time == 30.0
        assert job.algorithm_results[0].optimization_history == [{"trial": 1}]
        assert job.algorithm_results[0].status == TrainingStatus.COMPLETED
        
        assert job.algorithm_results[1].algorithm_name == "local_outlier_factor"
        assert job.algorithm_results[1].model_id == "model_2"
        assert job.algorithm_results[1].training_time == 25.0
        assert job.algorithm_results[1].optimization_history == []

    def test_duration_property_not_started(self):
        """Test duration property when job hasn't started."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert job.duration is None

    def test_duration_property_running(self):
        """Test duration property for running job."""
        started_at = datetime.utcnow() - timedelta(seconds=120)
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            started_at=started_at,
            status=TrainingStatus.RUNNING
        )
        
        duration = job.duration
        assert duration is not None
        assert duration >= 120  # At least 2 minutes

    def test_duration_property_completed(self):
        """Test duration property for completed job."""
        started_at = datetime.utcnow() - timedelta(seconds=180)
        completed_at = started_at + timedelta(seconds=120)
        
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            started_at=started_at,
            completed_at=completed_at,
            status=TrainingStatus.COMPLETED
        )
        
        assert job.duration == 120.0

    def test_is_running_property(self):
        """Test is_running property."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert not job.is_running
        
        job.status = TrainingStatus.RUNNING
        assert job.is_running
        
        job.status = TrainingStatus.COMPLETED
        assert not job.is_running

    def test_is_completed_property(self):
        """Test is_completed property."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert not job.is_completed
        
        job.status = TrainingStatus.RUNNING
        assert not job.is_completed
        
        job.status = TrainingStatus.COMPLETED
        assert job.is_completed
        
        job.status = TrainingStatus.FAILED
        assert job.is_completed
        
        job.status = TrainingStatus.CANCELLED
        assert job.is_completed

    def test_success_rate_property_no_results(self):
        """Test success_rate property with no algorithm results."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert job.success_rate == 0.0

    def test_success_rate_property_with_results(self):
        """Test success_rate property with mixed results."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        # Add successful results
        job.algorithm_results = [
            AlgorithmResult("algo1", status=TrainingStatus.COMPLETED),
            AlgorithmResult("algo2", status=TrainingStatus.COMPLETED),
            AlgorithmResult("algo3", status=TrainingStatus.FAILED),
            AlgorithmResult("algo4", status=TrainingStatus.PENDING)
        ]
        
        assert job.success_rate == 0.5  # 2 out of 4 successful

    def test_total_training_time_property(self):
        """Test total_training_time property."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        job.algorithm_results = [
            AlgorithmResult("algo1", training_time=30.0),
            AlgorithmResult("algo2", training_time=45.5),
            AlgorithmResult("algo3", training_time=22.3)
        ]
        
        assert job.total_training_time == 97.8

    def test_start_method(self):
        """Test start method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        start_time_before = datetime.utcnow()
        job.start()
        start_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.RUNNING
        assert start_time_before <= job.started_at <= start_time_after
        assert job.updated_at == job.started_at

    def test_complete_method(self):
        """Test complete method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        complete_time_before = datetime.utcnow()
        job.complete()
        complete_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.COMPLETED
        assert complete_time_before <= job.completed_at <= complete_time_after
        assert job.updated_at == job.completed_at
        assert job.progress == 100.0

    def test_fail_method(self):
        """Test fail method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        error_message = "Training failed due to insufficient data"
        error_details = {"code": "INSUFFICIENT_DATA", "samples": 10}
        
        fail_time_before = datetime.utcnow()
        job.fail(error_message, error_details)
        fail_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.FAILED
        assert fail_time_before <= job.completed_at <= fail_time_after
        assert job.updated_at == job.completed_at
        assert job.error_message == error_message
        assert job.error_details == error_details

    def test_fail_method_without_details(self):
        """Test fail method without error details."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        error_message = "Training failed"
        job.fail(error_message)
        
        assert job.status == TrainingStatus.FAILED
        assert job.error_message == error_message
        assert job.error_details == {}

    def test_cancel_method(self):
        """Test cancel method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        cancel_time_before = datetime.utcnow()
        job.cancel()
        cancel_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.CANCELLED
        assert cancel_time_before <= job.completed_at <= cancel_time_after
        assert job.updated_at == job.completed_at

    def test_pause_method(self):
        """Test pause method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", status=TrainingStatus.RUNNING)
        
        pause_time_before = datetime.utcnow()
        job.pause()
        pause_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.PAUSED
        assert pause_time_before <= job.updated_at <= pause_time_after

    def test_pause_method_not_running(self):
        """Test pause method when job is not running."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", status=TrainingStatus.PENDING)
        original_status = job.status
        original_updated_at = job.updated_at
        
        job.pause()
        
        # Status should not change if not running
        assert job.status == original_status
        assert job.updated_at == original_updated_at

    def test_resume_method(self):
        """Test resume method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", status=TrainingStatus.PAUSED)
        
        resume_time_before = datetime.utcnow()
        job.resume()
        resume_time_after = datetime.utcnow()
        
        assert job.status == TrainingStatus.RUNNING
        assert resume_time_before <= job.updated_at <= resume_time_after

    def test_resume_method_not_paused(self):
        """Test resume method when job is not paused."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", status=TrainingStatus.PENDING)
        original_status = job.status
        original_updated_at = job.updated_at
        
        job.resume()
        
        # Status should not change if not paused
        assert job.status == original_status
        assert job.updated_at == original_updated_at

    def test_update_progress_method(self):
        """Test update_progress method."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        update_time_before = datetime.utcnow()
        job.update_progress(75.5, "isolation_forest")
        update_time_after = datetime.utcnow()
        
        assert job.progress == 75.5
        assert job.current_algorithm == "isolation_forest"
        assert update_time_before <= job.updated_at <= update_time_after

    def test_update_progress_clamping(self):
        """Test progress value clamping."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        # Test values outside valid range
        job.update_progress(-10.0)
        assert job.progress == 0.0
        
        job.update_progress(150.0)
        assert job.progress == 100.0

    def test_update_progress_without_algorithm(self):
        """Test update_progress without specifying algorithm."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", current_algorithm="old_algo")
        
        job.update_progress(50.0)
        
        assert job.progress == 50.0
        assert job.current_algorithm == "old_algo"  # Should remain unchanged

    def test_add_algorithm_result_new(self):
        """Test adding new algorithm result."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        result = AlgorithmResult("isolation_forest", training_time=30.0)
        
        add_time_before = datetime.utcnow()
        job.add_algorithm_result(result)
        add_time_after = datetime.utcnow()
        
        assert len(job.algorithm_results) == 1
        assert job.algorithm_results[0] == result
        assert add_time_before <= job.updated_at <= add_time_after

    def test_add_algorithm_result_replace_existing(self):
        """Test adding algorithm result replaces existing one for same algorithm."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        # Add first result
        result1 = AlgorithmResult("isolation_forest", training_time=30.0)
        job.add_algorithm_result(result1)
        
        # Add second result for same algorithm
        result2 = AlgorithmResult("isolation_forest", training_time=45.0)
        job.add_algorithm_result(result2)
        
        assert len(job.algorithm_results) == 1
        assert job.algorithm_results[0] == result2
        assert job.algorithm_results[0].training_time == 45.0

    def test_add_algorithm_result_updates_best_model(self):
        """Test adding algorithm result updates best model when score is better."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        # Add first result
        metrics1 = TrainingMetrics(best_score=0.85)
        result1 = AlgorithmResult(
            "isolation_forest",
            model_id="model_1",
            metrics=metrics1,
            status=TrainingStatus.COMPLETED
        )
        job.add_algorithm_result(result1)
        
        assert job.best_model_id == "model_1"
        assert job.best_algorithm == "isolation_forest"
        assert job.final_metrics == metrics1
        
        # Add second result with better score
        metrics2 = TrainingMetrics(best_score=0.92)
        result2 = AlgorithmResult(
            "local_outlier_factor",
            model_id="model_2",
            metrics=metrics2,
            status=TrainingStatus.COMPLETED
        )
        job.add_algorithm_result(result2)
        
        assert job.best_model_id == "model_2"
        assert job.best_algorithm == "local_outlier_factor"
        assert job.final_metrics == metrics2

    def test_add_algorithm_result_no_update_worse_score(self):
        """Test adding algorithm result doesn't update best model when score is worse."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        # Add first result with high score
        metrics1 = TrainingMetrics(best_score=0.92)
        result1 = AlgorithmResult(
            "isolation_forest",
            model_id="model_1",
            metrics=metrics1,
            status=TrainingStatus.COMPLETED
        )
        job.add_algorithm_result(result1)
        
        # Add second result with worse score
        metrics2 = TrainingMetrics(best_score=0.85)
        result2 = AlgorithmResult(
            "local_outlier_factor",
            model_id="model_2",
            metrics=metrics2,
            status=TrainingStatus.COMPLETED
        )
        job.add_algorithm_result(result2)
        
        # Best model should remain the first one
        assert job.best_model_id == "model_1"
        assert job.best_algorithm == "isolation_forest"
        assert job.final_metrics == metrics1

    def test_add_algorithm_result_failed_status_no_update(self):
        """Test adding failed algorithm result doesn't update best model."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        metrics = TrainingMetrics(best_score=0.95)
        result = AlgorithmResult(
            "isolation_forest",
            model_id="model_1",
            metrics=metrics,
            status=TrainingStatus.FAILED
        )
        job.add_algorithm_result(result)
        
        assert job.best_model_id is None
        assert job.best_algorithm is None
        assert job.final_metrics is None

    def test_get_algorithm_result_found(self):
        """Test getting algorithm result that exists."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        result = AlgorithmResult("isolation_forest", training_time=30.0)
        job.add_algorithm_result(result)
        
        found_result = job.get_algorithm_result("isolation_forest")
        assert found_result == result

    def test_get_algorithm_result_not_found(self):
        """Test getting algorithm result that doesn't exist."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        found_result = job.get_algorithm_result("nonexistent_algorithm")
        assert found_result is None

    def test_get_successful_results(self):
        """Test getting successful algorithm results."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        result1 = AlgorithmResult("algo1", status=TrainingStatus.COMPLETED)
        result2 = AlgorithmResult("algo2", status=TrainingStatus.FAILED)
        result3 = AlgorithmResult("algo3", status=TrainingStatus.COMPLETED)
        result4 = AlgorithmResult("algo4", status=TrainingStatus.PENDING)
        
        job.algorithm_results = [result1, result2, result3, result4]
        
        successful = job.get_successful_results()
        assert len(successful) == 2
        assert result1 in successful
        assert result3 in successful

    def test_get_failed_results(self):
        """Test getting failed algorithm results."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        result1 = AlgorithmResult("algo1", status=TrainingStatus.COMPLETED)
        result2 = AlgorithmResult("algo2", status=TrainingStatus.FAILED)
        result3 = AlgorithmResult("algo3", status=TrainingStatus.FAILED)
        result4 = AlgorithmResult("algo4", status=TrainingStatus.PENDING)
        
        job.algorithm_results = [result1, result2, result3, result4]
        
        failed = job.get_failed_results()
        assert len(failed) == 2
        assert result2 in failed
        assert result3 in failed

    def test_add_tag(self):
        """Test adding tags."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        add_time_before = datetime.utcnow()
        job.add_tag("production")
        add_time_after = datetime.utcnow()
        
        assert "production" in job.tags
        assert add_time_before <= job.updated_at <= add_time_after
        
        # Adding same tag should not duplicate
        job.add_tag("production")
        assert job.tags.count("production") == 1

    def test_remove_tag(self):
        """Test removing tags."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456", tags=["production", "experiment"])
        
        remove_time_before = datetime.utcnow()
        job.remove_tag("production")
        remove_time_after = datetime.utcnow()
        
        assert "production" not in job.tags
        assert "experiment" in job.tags
        assert remove_time_before <= job.updated_at <= remove_time_after
        
        # Removing non-existent tag should not cause error
        job.remove_tag("nonexistent")

    def test_can_retry_true(self):
        """Test can_retry returns True when conditions are met."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            status=TrainingStatus.FAILED,
            retry_count=1,
            max_retries=3
        )
        
        assert job.can_retry() is True

    def test_can_retry_false_not_failed(self):
        """Test can_retry returns False when job is not failed."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            status=TrainingStatus.COMPLETED,
            retry_count=1,
            max_retries=3
        )
        
        assert job.can_retry() is False

    def test_can_retry_false_max_retries_reached(self):
        """Test can_retry returns False when max retries reached."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            status=TrainingStatus.FAILED,
            retry_count=3,
            max_retries=3
        )
        
        assert job.can_retry() is False

    def test_retry_success(self):
        """Test successful retry."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            status=TrainingStatus.FAILED,
            retry_count=1,
            max_retries=3,
            error_message="Test error",
            error_details={"code": "TEST_ERROR"}
        )
        
        retry_time_before = datetime.utcnow()
        job.retry()
        retry_time_after = datetime.utcnow()
        
        assert job.retry_count == 2
        assert job.status == TrainingStatus.PENDING
        assert job.error_message is None
        assert job.error_details == {}
        assert retry_time_before <= job.updated_at <= retry_time_after

    def test_retry_cannot_retry(self):
        """Test retry when cannot retry."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            status=TrainingStatus.COMPLETED,  # Not failed
            retry_count=1,
            max_retries=3
        )
        
        original_retry_count = job.retry_count
        original_status = job.status
        
        job.retry()
        
        # Should not change anything
        assert job.retry_count == original_retry_count
        assert job.status == original_status

    def test_estimate_completion_time_no_start(self):
        """Test estimate_completion_time when job hasn't started."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        assert job.estimate_completion_time() is None

    def test_estimate_completion_time_no_progress(self):
        """Test estimate_completion_time when progress is 0."""
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            started_at=datetime.utcnow(),
            progress=0.0
        )
        
        assert job.estimate_completion_time() is None

    def test_estimate_completion_time_with_progress(self):
        """Test estimate_completion_time with progress."""
        started_at = datetime.utcnow() - timedelta(seconds=300)  # 5 minutes ago
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            started_at=started_at,
            progress=25.0  # 25% complete
        )
        
        estimated = job.estimate_completion_time()
        
        assert estimated is not None
        assert isinstance(estimated, datetime)
        assert estimated == job.estimated_completion
        
        # Should be approximately 15 minutes from now (75% remaining / 25% per 5 min)
        expected_time = datetime.utcnow() + timedelta(seconds=900)  # 15 minutes
        assert abs((estimated - expected_time).total_seconds()) < 60  # Within 1 minute

    def test_to_dict_minimal(self):
        """Test to_dict with minimal job."""
        job = TrainingJob(id="job_123", dataset_id="dataset_456")
        
        result_dict = job.to_dict()
        
        assert result_dict["id"] == "job_123"
        assert result_dict["dataset_id"] == "dataset_456"
        assert result_dict["name"] is None
        assert result_dict["algorithms"] == []
        assert result_dict["priority"] == "normal"
        assert result_dict["status"] == "pending"
        assert result_dict["progress"] == 0.0
        assert result_dict["algorithm_results"] == []
        assert result_dict["success_rate"] == 0.0
        assert result_dict["total_training_time"] == 0.0

    def test_to_dict_complete(self):
        """Test to_dict with complete job."""
        created_at = datetime.utcnow()
        config = Mock()
        config.to_dict.return_value = {"param": "value"}
        
        job = TrainingJob(
            id="job_123",
            dataset_id="dataset_456",
            name="Test Job",
            config=config,
            created_at=created_at,
            tags=["test"],
            algorithms=["algo1"]
        )
        
        result_dict = job.to_dict()
        
        assert result_dict["name"] == "Test Job"
        assert result_dict["config"] == {"param": "value"}
        assert result_dict["created_at"] == created_at.isoformat()
        assert result_dict["tags"] == ["test"]
        assert result_dict["algorithms"] == ["algo1"]

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "id": "job_123",
            "dataset_id": "dataset_456",
            "status": "pending",
            "priority": "normal"
        }
        
        job = TrainingJob.from_dict(data)
        
        assert job.id == "job_123"
        assert job.dataset_id == "dataset_456"
        assert job.status == TrainingStatus.PENDING
        assert job.priority == TrainingPriority.NORMAL

    def test_from_dict_with_datetimes(self):
        """Test from_dict with datetime fields."""
        created_at = datetime.utcnow()
        started_at = created_at + timedelta(seconds=60)
        
        data = {
            "id": "job_123",
            "dataset_id": "dataset_456",
            "created_at": created_at.isoformat(),
            "started_at": started_at.isoformat(),
            "status": "running",
            "priority": "high"
        }
        
        job = TrainingJob.from_dict(data)
        
        assert job.created_at == created_at
        assert job.started_at == started_at
        assert job.status == TrainingStatus.RUNNING
        assert job.priority == TrainingPriority.HIGH

    def test_from_dict_removes_calculated_fields(self):
        """Test from_dict removes calculated fields."""
        data = {
            "id": "job_123",
            "dataset_id": "dataset_456",
            "duration": 120.0,
            "success_rate": 0.8,
            "total_training_time": 300.0,
            "status": "pending",
            "priority": "normal"
        }
        
        # Should not raise an error even with calculated fields
        job = TrainingJob.from_dict(data)
        
        assert job.id == "job_123"
        assert job.dataset_id == "dataset_456"


class TestTrainingStatusEnum:
    """Test cases for TrainingStatus enum."""

    def test_training_status_values(self):
        """Test all TrainingStatus enum values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"
        assert TrainingStatus.PAUSED.value == "paused"


class TestTrainingPriorityEnum:
    """Test cases for TrainingPriority enum."""

    def test_training_priority_values(self):
        """Test all TrainingPriority enum values."""
        assert TrainingPriority.LOW.value == "low"
        assert TrainingPriority.NORMAL.value == "normal"
        assert TrainingPriority.HIGH.value == "high"
        assert TrainingPriority.URGENT.value == "urgent"