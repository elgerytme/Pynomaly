"""Tests for data profile domain entities."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.packages.data_profiling.domain.entities.data_profile import (
    DataProfile, ProfilingStatus, ProfileId, DatasetId,
    SchemaProfile, QualityAssessment, ProfilingMetadata,
    ColumnProfile, DataType, CardinalityLevel, SemanticType,
    Pattern, ValueDistribution, StatisticalSummary,
    QualityIssue, ProfilingJob
)


class TestProfileId:
    """Test ProfileId value object."""
    
    def test_profile_id_creation(self):
        """Test ProfileId creation."""
        profile_id = ProfileId()
        assert profile_id.value is not None
        assert isinstance(profile_id.value, str)
        assert len(profile_id.value) == 36  # UUID length
    
    def test_profile_id_string_representation(self):
        """Test ProfileId string representation."""
        profile_id = ProfileId()
        assert str(profile_id) == profile_id.value
    
    def test_profile_id_uniqueness(self):
        """Test ProfileId uniqueness."""
        id1 = ProfileId()
        id2 = ProfileId()
        assert id1.value != id2.value


class TestDatasetId:
    """Test DatasetId value object."""
    
    def test_dataset_id_creation_default(self):
        """Test DatasetId creation with default value."""
        dataset_id = DatasetId()
        assert dataset_id.value is not None
        assert isinstance(dataset_id.value, str)
    
    def test_dataset_id_creation_with_value(self):
        """Test DatasetId creation with specified value."""
        test_value = "test_dataset_123"
        dataset_id = DatasetId(test_value)
        assert dataset_id.value == test_value
    
    def test_dataset_id_string_representation(self):
        """Test DatasetId string representation."""
        dataset_id = DatasetId("test_dataset")
        assert str(dataset_id) == "test_dataset"


class TestStatisticalSummary:
    """Test StatisticalSummary entity."""
    
    def test_statistical_summary_creation(self):
        """Test StatisticalSummary creation."""
        summary = StatisticalSummary(
            mean=10.5,
            median=9.0,
            std_dev=2.5,
            min_value=1.0,
            max_value=20.0
        )
        assert summary.mean == 10.5
        assert summary.median == 9.0
        assert summary.std_dev == 2.5
        assert summary.min_value == 1.0
        assert summary.max_value == 20.0
    
    def test_statistical_summary_defaults(self):
        """Test StatisticalSummary with default values."""
        summary = StatisticalSummary()
        assert summary.mean is None
        assert summary.median is None
        assert summary.std_dev is None


class TestValueDistribution:
    """Test ValueDistribution entity."""
    
    def test_value_distribution_creation(self):
        """Test ValueDistribution creation."""
        distribution = ValueDistribution(
            value_counts={"A": 10, "B": 5},
            unique_count=2,
            null_count=1,
            total_count=16,
            cardinality_level=CardinalityLevel.LOW
        )
        assert distribution.value_counts == {"A": 10, "B": 5}
        assert distribution.unique_count == 2
        assert distribution.null_count == 1
        assert distribution.total_count == 16
        assert distribution.cardinality_level == CardinalityLevel.LOW
    
    def test_value_distribution_defaults(self):
        """Test ValueDistribution with default values."""
        distribution = ValueDistribution()
        assert distribution.value_counts == {}
        assert distribution.unique_count == 0
        assert distribution.null_count == 0
        assert distribution.total_count == 0
        assert distribution.cardinality_level == CardinalityLevel.UNKNOWN


class TestPattern:
    """Test Pattern entity."""
    
    def test_pattern_creation(self):
        """Test Pattern creation."""
        pattern = Pattern(
            pattern_type="email",
            regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            frequency=15,
            percentage=75.0,
            examples=["user@example.com", "test@test.org"],
            confidence=0.9,
            description="Email address pattern"
        )
        assert pattern.pattern_type == "email"
        assert pattern.regex == r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert pattern.frequency == 15
        assert pattern.percentage == 75.0
        assert pattern.examples == ["user@example.com", "test@test.org"]
        assert pattern.confidence == 0.9
        assert pattern.description == "Email address pattern"
    
    def test_pattern_defaults(self):
        """Test Pattern with default values."""
        pattern = Pattern(
            pattern_type="test",
            regex=".*",
            frequency=10,
            percentage=50.0
        )
        assert pattern.examples == []
        assert pattern.confidence == 0.0
        assert pattern.description == ""


class TestColumnProfile:
    """Test ColumnProfile entity."""
    
    def test_column_profile_creation(self):
        """Test ColumnProfile creation."""
        profile = ColumnProfile(
            column_name="email",
            data_type=DataType.STRING,
            nullable=True,
            unique_count=100,
            null_count=5,
            total_count=105,
            completeness_ratio=0.95,
            cardinality=CardinalityLevel.HIGH,
            semantic_type=SemanticType.PII_EMAIL,
            quality_score=0.85
        )
        assert profile.column_name == "email"
        assert profile.data_type == DataType.STRING
        assert profile.nullable is True
        assert profile.unique_count == 100
        assert profile.null_count == 5
        assert profile.total_count == 105
        assert profile.completeness_ratio == 0.95
        assert profile.cardinality == CardinalityLevel.HIGH
        assert profile.semantic_type == SemanticType.PII_EMAIL
        assert profile.quality_score == 0.85
    
    def test_column_profile_uniqueness_ratio(self):
        """Test ColumnProfile uniqueness ratio calculation."""
        profile = ColumnProfile(
            column_name="test",
            data_type=DataType.STRING,
            unique_count=50,
            total_count=100
        )
        assert profile.uniqueness_ratio == 0.5
    
    def test_column_profile_uniqueness_ratio_zero_total(self):
        """Test ColumnProfile uniqueness ratio with zero total count."""
        profile = ColumnProfile(
            column_name="test",
            data_type=DataType.STRING,
            unique_count=0,
            total_count=0
        )
        assert profile.uniqueness_ratio == 0.0


class TestQualityIssue:
    """Test QualityIssue entity."""
    
    def test_quality_issue_creation(self):
        """Test QualityIssue creation."""
        issue = QualityIssue(
            issue_type="missing_values",
            severity="high",
            description="Column has high percentage of missing values",
            affected_columns=["email", "phone"],
            affected_rows=50,
            impact_percentage=25.0,
            suggested_actions=["Implement data imputation", "Review data collection"],
            rule_violated="completeness_threshold"
        )
        assert issue.issue_type == "missing_values"
        assert issue.severity == "high"
        assert issue.description == "Column has high percentage of missing values"
        assert issue.affected_columns == ["email", "phone"]
        assert issue.affected_rows == 50
        assert issue.impact_percentage == 25.0
        assert issue.suggested_actions == ["Implement data imputation", "Review data collection"]
        assert issue.rule_violated == "completeness_threshold"


class TestQualityAssessment:
    """Test QualityAssessment entity."""
    
    def test_quality_assessment_creation(self):
        """Test QualityAssessment creation."""
        assessment = QualityAssessment(
            overall_score=0.85,
            completeness_score=0.90,
            consistency_score=0.80,
            accuracy_score=0.85,
            validity_score=0.88,
            uniqueness_score=0.75,
            critical_issues=2,
            high_issues=5,
            medium_issues=10,
            low_issues=15
        )
        assert assessment.overall_score == 0.85
        assert assessment.completeness_score == 0.90
        assert assessment.consistency_score == 0.80
        assert assessment.accuracy_score == 0.85
        assert assessment.validity_score == 0.88
        assert assessment.uniqueness_score == 0.75
        assert assessment.critical_issues == 2
        assert assessment.high_issues == 5
        assert assessment.medium_issues == 10
        assert assessment.low_issues == 15
    
    def test_quality_assessment_total_issues(self):
        """Test QualityAssessment total issues calculation."""
        assessment = QualityAssessment(
            critical_issues=2,
            high_issues=5,
            medium_issues=10,
            low_issues=15
        )
        assert assessment.total_issues == 32


class TestSchemaProfile:
    """Test SchemaProfile entity."""
    
    def test_schema_profile_creation(self):
        """Test SchemaProfile creation."""
        columns = [
            ColumnProfile(
                column_name="id",
                data_type=DataType.INTEGER,
                nullable=False
            ),
            ColumnProfile(
                column_name="name",
                data_type=DataType.STRING,
                nullable=True
            )
        ]
        
        profile = SchemaProfile(
            total_columns=2,
            total_rows=1000,
            columns=columns,
            primary_keys=["id"],
            estimated_size_bytes=50000
        )
        
        assert profile.total_columns == 2
        assert profile.total_rows == 1000
        assert len(profile.columns) == 2
        assert profile.primary_keys == ["id"]
        assert profile.estimated_size_bytes == 50000


class TestProfilingMetadata:
    """Test ProfilingMetadata entity."""
    
    def test_profiling_metadata_creation(self):
        """Test ProfilingMetadata creation."""
        metadata = ProfilingMetadata(
            profiling_strategy="sample",
            sample_size=10000,
            sample_percentage=50.0,
            execution_time_seconds=15.5,
            memory_usage_mb=128.5,
            include_patterns=True,
            include_statistical_analysis=True,
            include_quality_assessment=True,
            engine_version="2.0.0",
            configuration={"advanced_analysis": True}
        )
        
        assert metadata.profiling_strategy == "sample"
        assert metadata.sample_size == 10000
        assert metadata.sample_percentage == 50.0
        assert metadata.execution_time_seconds == 15.5
        assert metadata.memory_usage_mb == 128.5
        assert metadata.include_patterns is True
        assert metadata.include_statistical_analysis is True
        assert metadata.include_quality_assessment is True
        assert metadata.engine_version == "2.0.0"
        assert metadata.configuration == {"advanced_analysis": True}


class TestDataProfile:
    """Test DataProfile entity."""
    
    def test_data_profile_creation(self):
        """Test DataProfile creation."""
        profile = DataProfile(
            source_type="postgresql",
            source_connection={"host": "localhost", "database": "test"}
        )
        
        assert profile.profile_id is not None
        assert profile.dataset_id is not None
        assert profile.source_type == "postgresql"
        assert profile.source_connection == {"host": "localhost", "database": "test"}
        assert profile.status == ProfilingStatus.PENDING
        assert profile.created_at is not None
        assert profile.started_at is None
        assert profile.completed_at is None
        assert profile.error_message is None
        assert profile.schema_profile is None
        assert profile.quality_assessment is None
        assert profile.profiling_metadata is None
    
    def test_data_profile_start_profiling(self):
        """Test DataProfile start profiling."""
        profile = DataProfile()
        profile.start_profiling()
        
        assert profile.status == ProfilingStatus.IN_PROGRESS
        assert profile.started_at is not None
    
    def test_data_profile_complete_profiling(self):
        """Test DataProfile complete profiling."""
        profile = DataProfile()
        profile.start_profiling()
        
        schema_profile = SchemaProfile()
        quality_assessment = QualityAssessment()
        metadata = ProfilingMetadata()
        
        profile.complete_profiling(schema_profile, quality_assessment, metadata)
        
        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.completed_at is not None
        assert profile.schema_profile == schema_profile
        assert profile.quality_assessment == quality_assessment
        assert profile.profiling_metadata == metadata
    
    def test_data_profile_fail_profiling(self):
        """Test DataProfile fail profiling."""
        profile = DataProfile()
        profile.start_profiling()
        
        error_message = "Test error"
        profile.fail_profiling(error_message)
        
        assert profile.status == ProfilingStatus.FAILED
        assert profile.completed_at is not None
        assert profile.error_message == error_message
    
    def test_data_profile_cancel_profiling(self):
        """Test DataProfile cancel profiling."""
        profile = DataProfile()
        profile.start_profiling()
        
        profile.cancel_profiling()
        
        assert profile.status == ProfilingStatus.CANCELLED
        assert profile.completed_at is not None
    
    def test_data_profile_execution_time(self):
        """Test DataProfile execution time calculation."""
        profile = DataProfile()
        profile.start_profiling()
        
        # Mock the completion time
        profile.completed_at = profile.started_at
        
        execution_time = profile.execution_time_seconds
        assert execution_time == 0.0
    
    def test_data_profile_execution_time_not_completed(self):
        """Test DataProfile execution time when not completed."""
        profile = DataProfile()
        assert profile.execution_time_seconds is None
    
    def test_data_profile_is_completed(self):
        """Test DataProfile is completed property."""
        profile = DataProfile()
        assert profile.is_completed is False
        
        profile.complete_profiling(SchemaProfile(), QualityAssessment(), ProfilingMetadata())
        assert profile.is_completed is True
    
    def test_data_profile_has_schema_profile(self):
        """Test DataProfile has schema profile property."""
        profile = DataProfile()
        assert profile.has_schema_profile is False
        
        profile.schema_profile = SchemaProfile()
        assert profile.has_schema_profile is True
    
    def test_data_profile_has_quality_assessment(self):
        """Test DataProfile has quality assessment property."""
        profile = DataProfile()
        assert profile.has_quality_assessment is False
        
        profile.quality_assessment = QualityAssessment()
        assert profile.has_quality_assessment is True


class TestProfilingJob:
    """Test ProfilingJob entity."""
    
    def test_profiling_job_creation(self):
        """Test ProfilingJob creation."""
        job = ProfilingJob(
            dataset_source={"type": "csv", "path": "/data/test.csv"},
            profiling_config={"include_patterns": True}
        )
        
        assert job.job_id is not None
        assert job.profile_id is not None
        assert job.dataset_source == {"type": "csv", "path": "/data/test.csv"}
        assert job.profiling_config == {"include_patterns": True}
        assert job.status == ProfilingStatus.PENDING
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.progress_percentage == 0.0
        assert job.current_step == ""
        assert job.result_profile is None
        assert job.error_details == []
        assert job.warnings == []
    
    def test_profiling_job_execution_time(self):
        """Test ProfilingJob execution time calculation."""
        job = ProfilingJob()
        job.started_at = datetime.now()
        job.completed_at = job.started_at
        
        execution_time = job.execution_time_seconds
        assert execution_time == 0.0
    
    def test_profiling_job_execution_time_not_completed(self):
        """Test ProfilingJob execution time when not completed."""
        job = ProfilingJob()
        assert job.execution_time_seconds is None
    
    def test_profiling_job_is_running(self):
        """Test ProfilingJob is running property."""
        job = ProfilingJob()
        assert job.is_running is False
        
        job.status = ProfilingStatus.IN_PROGRESS
        assert job.is_running is True
    
    def test_profiling_job_is_finished(self):
        """Test ProfilingJob is finished property."""
        job = ProfilingJob()
        assert job.is_finished is False
        
        job.status = ProfilingStatus.COMPLETED
        assert job.is_finished is True
        
        job.status = ProfilingStatus.FAILED
        assert job.is_finished is True
        
        job.status = ProfilingStatus.CANCELLED
        assert job.is_finished is True
        
        job.status = ProfilingStatus.IN_PROGRESS
        assert job.is_finished is False