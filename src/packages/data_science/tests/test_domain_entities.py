"""Tests for data science domain entities."""

import pytest
from datetime import datetime
from uuid import uuid4

from domain.entities import (
    DataScienceModel, AnalysisJob, StatisticalProfile,
    MachineLearningPipeline, FeatureStore
)
from domain.entities.data_science_model import ModelType, ModelStatus
from domain.entities.analysis_job import AnalysisType, JobStatus, Priority


class TestDataScienceModel:
    """Test cases for DataScienceModel entity."""
    
    def test_create_basic_model(self):
        """Test creating a basic data science model."""
        model = DataScienceModel(
            name="test_model",
            model_type=ModelType.MACHINE_LEARNING,
            algorithm="random_forest",
            version_number="1.0.0"
        )
        
        assert model.name == "test_model"
        assert model.model_type == ModelType.MACHINE_LEARNING
        assert model.algorithm == "random_forest"
        assert model.version_number == "1.0.0"
        assert model.status == ModelStatus.DRAFT
        assert model.features == []
        assert model.performance_metrics == {}
    
    def test_model_status_transitions(self):
        """Test model status transitions through lifecycle."""
        model = DataScienceModel(
            name="test_model",
            model_type=ModelType.MACHINE_LEARNING,
            algorithm="random_forest",
            version_number="1.0.0"
        )
        
        # Initial state
        assert model.status == ModelStatus.DRAFT
        assert model.is_trainable()
        assert not model.is_deployed()
        
        # Mark as training
        model.mark_as_training()
        assert model.status == ModelStatus.TRAINING
        
        # Mark as trained
        metrics = {"accuracy": 0.95, "precision": 0.92}
        model.mark_as_trained(metrics, training_duration=120.5)
        assert model.status == ModelStatus.TRAINED
        assert model.performance_metrics["accuracy"] == 0.95
        assert model.training_duration_seconds == 120.5
        assert model.trained_at is not None
        
        # Mark as validated
        validation_metrics = {"f1_score": 0.93}
        model.mark_as_validated(validation_metrics)
        assert model.status == ModelStatus.VALIDATED
        assert model.performance_metrics["f1_score"] == 0.93
        assert model.is_deployable()
        
        # Mark as deployed
        deployment_config = {"endpoint": "api/v1/model"}
        model.mark_as_deployed(deployment_config)
        assert model.status == ModelStatus.DEPLOYED
        assert model.is_deployed()
        assert model.deployed_at is not None
    
    def test_model_validation_errors(self):
        """Test model validation rules."""
        with pytest.raises(ValueError, match="Feature names must be unique"):
            DataScienceModel(
                name="test_model",
                model_type=ModelType.MACHINE_LEARNING,
                algorithm="random_forest",
                version_number="1.0.0",
                features=["feature1", "feature2", "feature1"]
            )
        
        with pytest.raises(ValueError, match="must be validated before deployment"):
            model = DataScienceModel(
                name="test_model",
                model_type=ModelType.MACHINE_LEARNING,
                algorithm="random_forest",
                version_number="1.0.0"
            )
            model.mark_as_deployed({})
    
    def test_model_metrics_management(self):
        """Test performance metrics management."""
        model = DataScienceModel(
            name="test_model",
            model_type=ModelType.MACHINE_LEARNING,
            algorithm="random_forest",
            version_number="1.0.0"
        )
        
        # Add metrics
        model.add_performance_metric("accuracy", 0.95)
        model.add_performance_metric("precision", 0.92)
        
        assert model.get_metric("accuracy") == 0.95
        assert model.get_metric("precision") == 0.92
        assert model.get_metric("nonexistent") is None
        
        # Calculate model score
        score = model.calculate_model_score()
        assert score == 0.935  # Average of 0.95 and 0.92
    
    def test_model_tags_management(self):
        """Test model tags management."""
        model = DataScienceModel(
            name="test_model",
            model_type=ModelType.MACHINE_LEARNING,
            algorithm="random_forest",
            version_number="1.0.0"
        )
        
        # Add tags
        model.add_tag("production")
        model.add_tag("v1")
        model.add_tag("PRODUCTION")  # Should be normalized to lowercase
        
        assert "production" in model.tags
        assert "v1" in model.tags
        assert len(model.tags) == 2  # No duplicates
        
        # Remove tag
        model.remove_tag("v1")
        assert "v1" not in model.tags
        assert len(model.tags) == 1


class TestAnalysisJob:
    """Test cases for AnalysisJob entity."""
    
    def test_create_basic_analysis_job(self):
        """Test creating a basic analysis job."""
        job = AnalysisJob(
            name="correlation_analysis",
            analysis_type=AnalysisType.CORRELATION_ANALYSIS,
            dataset_id="dataset_123",
            configuration={"method": "pearson"}
        )
        
        assert job.name == "correlation_analysis"
        assert job.analysis_type == AnalysisType.CORRELATION_ANALYSIS
        assert job.dataset_id == "dataset_123"
        assert job.status == JobStatus.PENDING
        assert job.priority == Priority.NORMAL
    
    def test_job_execution_lifecycle(self):
        """Test job execution lifecycle."""
        job = AnalysisJob(
            name="statistical_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_id="dataset_123",
            configuration={}
        )
        
        # Queue job
        job.queue_job()
        assert job.status == JobStatus.QUEUED
        assert job.queued_at is not None
        
        # Start execution
        job.start_execution()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        
        # Complete job
        results = {"mean": 10.5, "std": 2.3}
        job.complete_job(results)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.results == results
        assert job.get_execution_duration() > 0
    
    def test_job_failure_handling(self):
        """Test job failure handling."""
        job = AnalysisJob(
            name="failing_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_id="dataset_123",
            configuration={}
        )
        
        job.start_execution()
        
        # Fail job
        error_msg = "Division by zero in statistical calculation"
        job.fail_job(error_msg)
        
        assert job.status == JobStatus.FAILED
        assert job.failed_at is not None
        assert job.error_message == error_msg
        assert not job.can_be_restarted()
    
    def test_job_priority_and_scheduling(self):
        """Test job priority and scheduling features."""
        job = AnalysisJob(
            name="urgent_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_id="dataset_123",
            configuration={},
            priority=Priority.URGENT
        )
        
        assert job.priority == Priority.URGENT
        assert job.estimated_duration is None
        
        # Set estimated duration
        job.set_estimated_duration(300)  # 5 minutes
        assert job.estimated_duration == 300
        
        # Reschedule job
        future_time = datetime.utcnow()
        job.reschedule_job(future_time)
        assert job.scheduled_at == future_time


class TestStatisticalProfile:
    """Test cases for StatisticalProfile entity."""
    
    def test_create_statistical_profile(self):
        """Test creating a statistical profile."""
        profile = StatisticalProfile(
            dataset_id="dataset_123",
            profile_name="basic_statistics",
            feature_count=10,
            row_count=1000
        )
        
        assert profile.dataset_id == "dataset_123"
        assert profile.profile_name == "basic_statistics"
        assert profile.feature_count == 10
        assert profile.row_count == 1000
        assert profile.statistical_metrics == {}
        assert profile.correlation_matrix is None
    
    def test_profile_metrics_management(self):
        """Test adding and retrieving statistical metrics."""
        profile = StatisticalProfile(
            dataset_id="dataset_123",
            profile_name="detailed_statistics",
            feature_count=5,
            row_count=500
        )
        
        # Add feature metrics
        feature_stats = {
            "mean": 10.5,
            "std": 2.3,
            "min": 5.0,
            "max": 18.0
        }
        profile.add_feature_statistics("feature1", feature_stats)
        
        assert "feature1" in profile.statistical_metrics
        assert profile.statistical_metrics["feature1"] == feature_stats
        
        # Get feature statistics
        retrieved_stats = profile.get_feature_statistics("feature1")
        assert retrieved_stats == feature_stats
        
        # Get non-existent feature
        assert profile.get_feature_statistics("nonexistent") is None
    
    def test_profile_quality_assessment(self):
        """Test profile quality assessment."""
        profile = StatisticalProfile(
            dataset_id="dataset_123",
            profile_name="quality_test",
            feature_count=3,
            row_count=100
        )
        
        # Mark as high quality
        profile.quality_score = 0.95
        profile.completeness = 0.98
        profile.validity = 0.92
        
        assert profile.is_high_quality()
        assert profile.is_complete()
        
        # Mark as low quality
        profile.quality_score = 0.45
        assert not profile.is_high_quality()


if __name__ == "__main__":
    pytest.main([__file__])