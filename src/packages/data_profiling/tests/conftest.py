import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import Mock
from uuid import uuid4
from datetime import datetime

from data_profiling.domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus, DataType,
    CardinalityLevel, PatternType, QualityIssueType, ValueDistribution,
    StatisticalSummary, Pattern, QualityIssue, ColumnProfile, SchemaProfile,
    QualityAssessment, ProfilingMetadata
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for profiling testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'email': [f'user{i}@example.com' for i in range(1, 1001)],
        'phone': [f'+1-555-{i:04d}' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR'], 1000),
        'is_active': np.random.choice([True, False], 1000),
        'created_at': pd.date_range('2020-01-01', periods=1000, freq='H'),
        'notes': [None if i % 10 == 0 else f'Note for user {i}' for i in range(1, 1001)]
    })


@pytest.fixture
def messy_dataframe():
    """Create a messy DataFrame with quality issues for testing."""
    return pd.DataFrame({
        'id': [1, 2, 2, 4, 5, None, 7, 8, 9, 10],  # Duplicates and nulls
        'email': [
            'valid@example.com',
            'invalid-email',
            'another@valid.com',
            '',
            None,
            'no-at-sign.com',
            'valid2@example.com',
            'UPPERCASE@EXAMPLE.COM',
            'valid3@example.com',
            'valid4@example.com'
        ],
        'phone': [
            '+1-555-1234',
            '555-5678',
            '(555) 901-2345',
            'invalid-phone',
            None,
            '555.123.4567',
            '+1 555 890 1234',
            '15551234567',
            'not-a-phone',
            '+1-555-9999'
        ],
        'age': [-5, 25, 30, 150, None, 35, 40, 45, 50, 55],  # Invalid ages
        'category': ['A', 'B', 'C', 'UNKNOWN', None, 'A', 'B', 'C', 'D', 'A']
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def profile_id():
    """Create a profile ID."""
    return ProfileId(value=uuid4())


@pytest.fixture
def dataset_id():
    """Create a dataset ID."""
    return DatasetId(value=uuid4())


@pytest.fixture
def sample_value_distribution():
    """Create a sample value distribution."""
    return ValueDistribution(
        unique_count=950,
        null_count=50,
        total_count=1000,
        completeness_ratio=0.95,
        top_values={
            "value1": 100,
            "value2": 80,
            "value3": 70
        }
    )


@pytest.fixture
def sample_statistical_summary():
    """Create a sample statistical summary."""
    return StatisticalSummary(
        min_value=18.0,
        max_value=79.0,
        mean=48.5,
        median=47.0,
        std_dev=15.2,
        quartiles=[33.0, 47.0, 63.0]
    )


@pytest.fixture
def email_pattern():
    """Create an email pattern."""
    return Pattern(
        pattern_type=PatternType.EMAIL,
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        frequency=950,
        percentage=95.0,
        examples=['user1@example.com', 'user2@example.com'],
        confidence=0.95
    )


@pytest.fixture
def phone_pattern():
    """Create a phone pattern."""
    return Pattern(
        pattern_type=PatternType.PHONE,
        regex=r'^\+1-555-\d{4}$',
        frequency=900,
        percentage=90.0,
        examples=['+1-555-0001', '+1-555-0002'],
        confidence=0.90
    )


@pytest.fixture
def quality_issue():
    """Create a quality issue."""
    return QualityIssue(
        issue_type=QualityIssueType.MISSING_VALUES,
        severity="medium",
        description="10% of values are missing",
        affected_rows=100,
        affected_percentage=10.0,
        examples=["Row 5", "Row 15", "Row 25"],
        suggested_action="Consider imputation or removing rows with missing values"
    )


@pytest.fixture
def sample_column_profile(sample_value_distribution, email_pattern, quality_issue):
    """Create a sample column profile."""
    return ColumnProfile(
        column_name="email",
        data_type=DataType.STRING,
        inferred_type=DataType.STRING,
        nullable=True,
        distribution=sample_value_distribution,
        cardinality=CardinalityLevel.VERY_HIGH,
        patterns=[email_pattern],
        quality_score=0.85,
        quality_issues=[quality_issue],
        semantic_type="email",
        business_meaning="User email address"
    )


@pytest.fixture
def sample_schema_profile(sample_column_profile):
    """Create a sample schema profile."""
    return SchemaProfile(
        table_name="users",
        total_columns=10,
        total_rows=1000,
        columns=[sample_column_profile],
        primary_keys=["id"],
        foreign_keys={"department_id": "departments.id"},
        unique_constraints=[["email"]],
        check_constraints=["age > 0"],
        estimated_size_bytes=1024000,
        compression_ratio=0.7
    )


@pytest.fixture
def sample_quality_assessment():
    """Create a sample quality assessment."""
    return QualityAssessment(
        overall_score=0.82,
        completeness_score=0.90,
        consistency_score=0.85,
        accuracy_score=0.80,
        validity_score=0.75,
        uniqueness_score=0.95,
        dimension_weights={
            "completeness": 0.2,
            "consistency": 0.2,
            "accuracy": 0.2,
            "validity": 0.2,
            "uniqueness": 0.2
        },
        critical_issues=0,
        high_issues=2,
        medium_issues=5,
        low_issues=10,
        recommendations=[
            "Improve email validation",
            "Address missing values in notes column"
        ]
    )


@pytest.fixture
def sample_profiling_metadata():
    """Create sample profiling metadata."""
    return ProfilingMetadata(
        profiling_strategy="sample",
        sample_size=1000,
        sample_percentage=10.0,
        execution_time_seconds=45.5,
        memory_usage_mb=256.0,
        include_patterns=True,
        include_statistical_analysis=True,
        include_quality_assessment=True
    )


@pytest.fixture
def sample_data_profile(
    profile_id,
    dataset_id,
    sample_schema_profile,
    sample_quality_assessment,
    sample_profiling_metadata
):
    """Create a sample data profile."""
    return DataProfile(
        profile_id=profile_id,
        dataset_id=dataset_id,
        status=ProfilingStatus.PENDING,
        source_type="database",
        source_connection={"host": "localhost", "database": "test"},
        source_query="SELECT * FROM users"
    )


@pytest.fixture
def completed_data_profile(
    sample_data_profile,
    sample_schema_profile,
    sample_quality_assessment,
    sample_profiling_metadata
):
    """Create a completed data profile."""
    profile = sample_data_profile
    profile.complete_profiling(
        schema_profile=sample_schema_profile,
        quality_assessment=sample_quality_assessment,
        metadata=sample_profiling_metadata
    )
    return profile


@pytest.fixture
def mock_data_profile_repository():
    """Create a mock data profile repository."""
    return Mock()


@pytest.fixture
def mock_database_connection_service():
    """Create a mock database connection service."""
    return Mock()


@pytest.fixture
def mock_statistical_engine():
    """Create a mock statistical engine."""
    return Mock()


@pytest.fixture
def mock_pattern_recognition_engine():
    """Create a mock pattern recognition engine."""
    return Mock()


@pytest.fixture
def profiling_config():
    """Create profiling configuration."""
    return {
        "strategy": "sample",
        "sample_percentage": 10.0,
        "include_patterns": True,
        "include_statistics": True,
        "include_quality": True,
        "parallel_processing": True
    }