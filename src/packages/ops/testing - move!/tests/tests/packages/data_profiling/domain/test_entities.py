import pytest
import pandas as pd
from datetime import datetime
from uuid import UUID

from src.packages.data_profiling.domain.entities.data_profile import (
    DataProfile, SchemaProfile, ColumnProfile, ValueDistribution,
    StatisticalSummary, ProfilingMetadata, QualityAssessment,
    ProfileId, DatasetId, DataType, CardinalityLevel, SemanticType,
    ProfilingStatus, Pattern, PatternType
)


class TestProfileId:
    """Test ProfileId value object."""
    
    def test_profile_id_creation(self):
        """Test ProfileId creation."""
        profile_id = ProfileId()
        assert isinstance(profile_id.value, str)
        assert len(profile_id.value) > 0
        
    def test_profile_id_with_value(self):
        """Test ProfileId creation with specific value."""
        test_id = "test-profile-123"
        profile_id = ProfileId(test_id)
        assert profile_id.value == test_id
        
    def test_profile_id_equality(self):
        """Test ProfileId equality."""
        id1 = ProfileId("same-id")
        id2 = ProfileId("same-id")
        id3 = ProfileId("different-id")
        
        assert id1 == id2
        assert id1 != id3


class TestDatasetId:
    """Test DatasetId value object."""
    
    def test_dataset_id_creation(self):
        """Test DatasetId creation."""
        dataset_id = DatasetId()
        assert isinstance(dataset_id.value, str)
        assert len(dataset_id.value) > 0
        
    def test_dataset_id_with_value(self):
        """Test DatasetId creation with specific value."""
        test_id = "test-dataset-456"
        dataset_id = DatasetId(test_id)
        assert dataset_id.value == test_id


class TestStatisticalSummary:
    """Test StatisticalSummary entity."""
    
    def test_statistical_summary_creation(self):
        """Test StatisticalSummary creation with all fields."""
        summary = StatisticalSummary(
            min_value=1.0,
            max_value=100.0,
            mean=50.0,
            median=45.0,
            std_dev=15.5,
            quartiles=[25.0, 45.0, 75.0]
        )
        
        assert summary.min_value == 1.0
        assert summary.max_value == 100.0
        assert summary.mean == 50.0
        assert summary.median == 45.0
        assert summary.std_dev == 15.5
        assert summary.quartiles == [25.0, 45.0, 75.0]
        
    def test_statistical_summary_defaults(self):
        """Test StatisticalSummary creation with defaults."""
        summary = StatisticalSummary()
        
        assert summary.min_value is None
        assert summary.max_value is None
        assert summary.mean is None
        assert summary.median is None
        assert summary.std_dev is None
        assert summary.quartiles == []


class TestValueDistribution:
    """Test ValueDistribution entity."""
    
    def test_value_distribution_creation(self):
        """Test ValueDistribution creation."""
        distribution = ValueDistribution(
            unique_count=50,
            null_count=5,
            most_frequent_values={"A": 10, "B": 8, "C": 7},
            frequency_distribution={"A": 0.2, "B": 0.16, "C": 0.14}
        )
        
        assert distribution.unique_count == 50
        assert distribution.null_count == 5
        assert distribution.most_frequent_values == {"A": 10, "B": 8, "C": 7}
        assert distribution.frequency_distribution == {"A": 0.2, "B": 0.16, "C": 0.14}


class TestPattern:
    """Test Pattern entity."""
    
    def test_pattern_creation(self):
        """Test Pattern creation."""
        pattern = Pattern(
            pattern_type=PatternType.EMAIL,
            pattern_regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            confidence_score=0.95,
            examples=["test@example.com", "user@domain.org"],
            match_count=150
        )
        
        assert pattern.pattern_type == PatternType.EMAIL
        assert pattern.pattern_regex == r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert pattern.confidence_score == 0.95
        assert pattern.examples == ["test@example.com", "user@domain.org"]
        assert pattern.match_count == 150


class TestColumnProfile:
    """Test ColumnProfile entity."""
    
    def test_column_profile_creation(self):
        """Test ColumnProfile creation with all components."""
        statistical_summary = StatisticalSummary(min_value=1.0, max_value=100.0)
        value_distribution = ValueDistribution(unique_count=50, null_count=5)
        patterns = [Pattern(PatternType.NUMERIC, r'\\d+', 0.8, ["123", "456"], 100)]
        
        column_profile = ColumnProfile(
            column_name="test_column",
            data_type=DataType.INTEGER,
            cardinality_level=CardinalityLevel.MEDIUM,
            semantic_type=SemanticType.IDENTIFIER,
            statistical_summary=statistical_summary,
            value_distribution=value_distribution,
            patterns=patterns,
            is_nullable=True,
            is_primary_key=False,
            is_foreign_key=False
        )
        
        assert column_profile.column_name == "test_column"
        assert column_profile.data_type == DataType.INTEGER
        assert column_profile.cardinality_level == CardinalityLevel.MEDIUM
        assert column_profile.semantic_type == SemanticType.IDENTIFIER
        assert column_profile.statistical_summary == statistical_summary
        assert column_profile.value_distribution == value_distribution
        assert column_profile.patterns == patterns
        assert column_profile.is_nullable is True
        assert column_profile.is_primary_key is False
        assert column_profile.is_foreign_key is False


class TestSchemaProfile:
    """Test SchemaProfile entity."""
    
    def test_schema_profile_creation(self):
        """Test SchemaProfile creation."""
        column_profile = ColumnProfile(
            column_name="test_col",
            data_type=DataType.STRING,
            cardinality_level=CardinalityLevel.LOW
        )
        
        schema_profile = SchemaProfile(
            total_tables=1,
            total_columns=1,
            total_rows=1000,
            columns=[column_profile]
        )
        
        assert schema_profile.total_tables == 1
        assert schema_profile.total_columns == 1
        assert schema_profile.total_rows == 1000
        assert len(schema_profile.columns) == 1
        assert schema_profile.columns[0] == column_profile


class TestQualityAssessment:
    """Test QualityAssessment entity."""
    
    def test_quality_assessment_creation(self):
        """Test QualityAssessment creation."""
        quality_assessment = QualityAssessment(
            overall_score=0.85,
            completeness_score=0.90,
            consistency_score=0.80,
            accuracy_score=0.85,
            validity_score=0.95,
            uniqueness_score=0.75,
            timeliness_score=0.88
        )
        
        assert quality_assessment.overall_score == 0.85
        assert quality_assessment.completeness_score == 0.90
        assert quality_assessment.consistency_score == 0.80
        assert quality_assessment.accuracy_score == 0.85
        assert quality_assessment.validity_score == 0.95
        assert quality_assessment.uniqueness_score == 0.75
        assert quality_assessment.timeliness_score == 0.88


class TestProfilingMetadata:
    """Test ProfilingMetadata entity."""
    
    def test_profiling_metadata_creation(self):
        """Test ProfilingMetadata creation."""
        metadata = ProfilingMetadata(
            profiling_strategy="intelligent_sampling",
            sample_size=10000,
            sample_percentage=25.0,
            execution_time_seconds=45.5,
            memory_usage_mb=128.0,
            include_patterns=True,
            include_statistical_analysis=True,
            include_quality_assessment=True
        )
        
        assert metadata.profiling_strategy == "intelligent_sampling"
        assert metadata.sample_size == 10000
        assert metadata.sample_percentage == 25.0
        assert metadata.execution_time_seconds == 45.5
        assert metadata.memory_usage_mb == 128.0
        assert metadata.include_patterns is True
        assert metadata.include_statistical_analysis is True
        assert metadata.include_quality_assessment is True


class TestDataProfile:
    """Test DataProfile entity."""
    
    def test_data_profile_creation(self):
        """Test DataProfile creation."""
        profile_id = ProfileId()
        dataset_id = DatasetId()
        
        profile = DataProfile(
            profile_id=profile_id,
            dataset_id=dataset_id,
            source_type="csv",
            source_connection={"path": "/data/test.csv"}
        )
        
        assert profile.profile_id == profile_id
        assert profile.dataset_id == dataset_id
        assert profile.source_type == "csv"
        assert profile.source_connection == {"path": "/data/test.csv"}
        assert profile.status == ProfilingStatus.PENDING
        assert profile.schema_profile is None
        assert profile.quality_assessment is None
        assert profile.metadata is None
        
    def test_data_profile_lifecycle(self):
        """Test DataProfile status lifecycle."""
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=DatasetId()
        )
        
        # Start profiling
        profile.start_profiling()
        assert profile.status == ProfilingStatus.IN_PROGRESS
        assert profile.started_at is not None
        
        # Complete profiling
        schema_profile = SchemaProfile(
            total_tables=1,
            total_columns=2,
            total_rows=100,
            columns=[]
        )
        quality_assessment = QualityAssessment(overall_score=0.8)
        metadata = ProfilingMetadata(execution_time_seconds=30.0)
        
        profile.complete_profiling(schema_profile, quality_assessment, metadata)
        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.completed_at is not None
        assert profile.schema_profile == schema_profile
        assert profile.quality_assessment == quality_assessment
        assert profile.metadata == metadata
        
    def test_data_profile_failure(self):
        """Test DataProfile failure handling."""
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=DatasetId()
        )
        
        profile.start_profiling()
        profile.fail_profiling("Test error message")
        
        assert profile.status == ProfilingStatus.FAILED
        assert profile.error_message == "Test error message"
        assert profile.completed_at is not None


class TestEnums:
    """Test enum classes."""
    
    def test_data_type_enum(self):
        """Test DataType enum."""
        assert DataType.STRING == "string"
        assert DataType.INTEGER == "integer"
        assert DataType.FLOAT == "float"
        assert DataType.BOOLEAN == "boolean"
        assert DataType.DATETIME == "datetime"
        assert DataType.UNKNOWN == "unknown"
        
    def test_cardinality_level_enum(self):
        """Test CardinalityLevel enum."""
        assert CardinalityLevel.LOW == "low"
        assert CardinalityLevel.MEDIUM == "medium"
        assert CardinalityLevel.HIGH == "high"
        
    def test_semantic_type_enum(self):
        """Test SemanticType enum."""
        assert SemanticType.IDENTIFIER == "identifier"
        assert SemanticType.PII_EMAIL == "pii_email"
        assert SemanticType.PII_PHONE == "pii_phone"
        assert SemanticType.FINANCIAL_AMOUNT == "financial_amount"
        
    def test_profiling_status_enum(self):
        """Test ProfilingStatus enum."""
        assert ProfilingStatus.PENDING == "pending"
        assert ProfilingStatus.IN_PROGRESS == "in_progress"
        assert ProfilingStatus.COMPLETED == "completed"
        assert ProfilingStatus.FAILED == "failed"
        
    def test_pattern_type_enum(self):
        """Test PatternType enum."""
        assert PatternType.EMAIL == "email"
        assert PatternType.PHONE == "phone"
        assert PatternType.URL == "url"
        assert PatternType.NUMERIC == "numeric"