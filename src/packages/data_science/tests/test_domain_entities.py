"""Tests for data science domain entities."""

import pytest
from datetime import datetime
from uuid import uuid4

from packages.data_science.domain.entities.data_science_model import (
    DataScienceModel, ModelType, ModelStatus
)
from packages.data_science.domain.entities.analysis_job import (
    AnalysisJob, AnalysisType, JobStatus, Priority
)
from packages.data_science.domain.entities.statistical_profile import (
    StatisticalProfile, ProfileType, ProfileScope
)
from packages.data_science.domain.entities.machine_learning_pipeline import MachineLearningPipeline
from packages.data_science.domain.entities.feature_store import FeatureStore


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
            dataset_ids=["dataset_123"],
            parameters={"method": "pearson"}
        )
        
        assert job.name == "correlation_analysis"
        assert job.analysis_type == AnalysisType.CORRELATION_ANALYSIS
        assert job.dataset_ids == ["dataset_123"]
        assert job.status == JobStatus.PENDING
        assert job.priority == Priority.NORMAL
    
    def test_job_execution_lifecycle(self):
        """Test job execution lifecycle."""
        job = AnalysisJob(
            name="statistical_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_ids=["dataset_123"],
            parameters={}
        )
        
        # Start execution
        job.start_execution("executor_123")
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert job.executor_id == "executor_123"
        
        # Update progress
        job.update_progress(50.0, {"intermediate": "result"})
        assert job.progress_percentage == 50.0
        
        # Complete job
        job.complete_successfully("s3://results/job_123", {"mean": 10.5, "std": 2.3})
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.results_uri == "s3://results/job_123"
        assert job.actual_duration_seconds is not None
    
    def test_job_failure_handling(self):
        """Test job failure handling."""
        job = AnalysisJob(
            name="failing_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_ids=["dataset_123"],
            parameters={}
        )
        
        job.start_execution("executor_123")
        
        # Fail job
        error_msg = "Division by zero in statistical calculation"
        job.fail_with_error(error_msg, should_retry=False)
        
        assert job.status == JobStatus.FAILED
        assert job.error_message == error_msg
    
    def test_job_priority_and_scheduling(self):
        """Test job priority and scheduling features."""
        job = AnalysisJob(
            name="urgent_analysis",
            analysis_type=AnalysisType.STATISTICAL,
            dataset_ids=["dataset_123"],
            parameters={},
            priority=Priority.URGENT
        )
        
        assert job.priority == Priority.URGENT
        assert job.estimated_duration_seconds is None
        
        # Set estimated duration
        job.estimated_duration_seconds = 300  # 5 minutes
        assert job.estimated_duration_seconds == 300
        
        # Schedule job
        future_time = datetime.utcnow()
        job.scheduled_at = future_time
        assert job.scheduled_at == future_time


class TestStatisticalProfile:
    """Test cases for StatisticalProfile entity."""
    
    def test_create_statistical_profile(self):
        """Test creating a statistical profile."""
        profile = StatisticalProfile(
            name="basic_statistics",
            profile_type=ProfileType.DESCRIPTIVE,
            scope=ProfileScope.DATASET,
            dataset_id="dataset_123",
            sample_size=1000,
            feature_names=["feature1", "feature2", "feature3"]
        )
        
        assert profile.name == "basic_statistics"
        assert profile.profile_type == ProfileType.DESCRIPTIVE
        assert profile.scope == ProfileScope.DATASET
        assert profile.dataset_id == "dataset_123"
        assert profile.sample_size == 1000
        assert len(profile.feature_names) == 3
        assert profile.descriptive_statistics == {}
        assert profile.correlation_analysis == {}
    
    def test_profile_metrics_management(self):
        """Test adding and retrieving statistical metrics."""
        profile = StatisticalProfile(
            name="detailed_statistics",
            profile_type=ProfileType.DESCRIPTIVE,
            scope=ProfileScope.FEATURE,
            dataset_id="dataset_123",
            sample_size=500,
            feature_names=["feature1"]
        )
        
        # Add descriptive statistics
        profile.add_descriptive_statistic("mean", 10.5)
        profile.add_descriptive_statistic("std", 2.3)
        
        assert profile.descriptive_statistics["mean"] == 10.5
        assert profile.descriptive_statistics["std"] == 2.3
        
        # Add hypothesis test
        profile.add_hypothesis_test("normality_test", 2.45, 0.03, {"test_type": "shapiro"})
        
        assert "normality_test" in profile.hypothesis_tests
        assert profile.hypothesis_tests["normality_test"]["statistic"] == 2.45
        assert profile.hypothesis_tests["normality_test"]["p_value"] == 0.03
        assert profile.hypothesis_tests["normality_test"]["significant"] == True
    
    def test_profile_quality_assessment(self):
        """Test profile quality assessment."""
        profile = StatisticalProfile(
            name="quality_test",
            profile_type=ProfileType.DESCRIPTIVE,
            scope=ProfileScope.DATASET,
            dataset_id="dataset_123",
            sample_size=1000,  # Large sample for good quality
            feature_names=["feature1", "feature2", "feature3"]
        )
        
        # Add comprehensive analysis to increase completeness
        profile.add_descriptive_statistic("mean", 10.5)
        profile.add_hypothesis_test("normality", 2.45, 0.8, {})  # Non-significant
        profile.add_correlation("feature1", "feature2", 0.85, 0.001)
        profile.validate_assumption("normality", True)
        
        # Calculate quality score
        quality_score = profile.calculate_quality_score()
        assert quality_score > 0.5  # Should be reasonably high
        assert profile.quality_score == quality_score
        
        # Check completeness increased
        assert profile.completeness_percentage > 0


if __name__ == "__main__":
    pytest.main([__file__])