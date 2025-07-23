"""Pytest configuration for data quality package testing."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock

# Import shared test utilities
from test_utilities.fixtures import *
from test_utilities.factories import *
from test_utilities.helpers import *
from test_utilities.markers import *

from data_quality.domain.entities.data_quality_check import (
    DataQualityCheck, CheckType, CheckStatus, CheckResult, CheckSeverity
)
from data_quality.domain.entities.data_quality_rule import (
    DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator
)
from data_quality.domain.entities.data_profile import (
    DataProfile, ColumnProfile, DataType, ProfileStatistics, ProfileStatus
)


@pytest.fixture
def sample_check_result():
    """Sample check result for testing."""
    return CheckResult(
        dataset_name="users",
        column_name="email",
        passed=True,
        score=0.95,
        total_records=1000,
        passed_records=950,
        failed_records=50,
        executed_at=datetime(2024, 1, 15, 10, 0, 0),
        execution_time_ms=150.0,
        severity=CheckSeverity.INFO,
        message="Email validation check passed"
    )


@pytest.fixture
def sample_data_quality_check():
    """Sample data quality check for testing."""
    return DataQualityCheck(
        name="Email Validation",
        description="Validate email format in users table",
        check_type=CheckType.VALIDITY,
        dataset_name="users",
        column_name="email",
        expression=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        threshold=0.95,
        created_by="data_engineer"
    )


@pytest.fixture
def sample_rule_condition():
    """Sample rule condition for testing."""
    return RuleCondition(
        column_name="age",
        operator=RuleOperator.BETWEEN,
        min_value=18,
        max_value=120,
        description="Age must be between 18 and 120"
    )


@pytest.fixture
def sample_data_quality_rule(sample_rule_condition):
    """Sample data quality rule for testing."""
    rule = DataQualityRule(
        name="Age Range Validation",
        description="Validate age is within acceptable range",
        rule_type=RuleType.RANGE,
        severity=RuleSeverity.ERROR,
        dataset_name="users",
        violation_threshold=0.05,
        created_by="data_engineer"
    )
    rule.add_condition(sample_rule_condition)
    return rule


@pytest.fixture
def sample_profile_statistics():
    """Sample profile statistics for testing."""
    return ProfileStatistics(
        total_count=1000,
        null_count=50,
        distinct_count=950,
        duplicate_count=50,
        min_length=5,
        max_length=50,
        avg_length=25.5,
        min_value=18,
        max_value=85,
        mean=45.2,
        median=43.0,
        std_dev=12.5
    )


@pytest.fixture
def sample_column_profile(sample_profile_statistics):
    """Sample column profile for testing."""
    profile = ColumnProfile(
        column_name="age",
        data_type=DataType.INTEGER,
        position=2,
        is_nullable=False,
        statistics=sample_profile_statistics,
        quality_score=0.95
    )
    
    # Add some top values
    profile.add_top_value(25, 50)
    profile.add_top_value(30, 45)
    profile.add_top_value(35, 40)
    
    return profile


@pytest.fixture
def sample_data_profile(sample_column_profile):
    """Sample data profile for testing."""
    profile = DataProfile(
        dataset_name="users",
        table_name="user_accounts",
        schema_name="public",
        total_rows=1000,
        total_columns=5,
        overall_quality_score=0.92,
        created_by="data_engineer"
    )
    
    profile.add_column_profile(sample_column_profile)
    
    # Add additional column profiles
    email_profile = ColumnProfile(
        column_name="email",
        data_type=DataType.STRING,
        position=1,
        is_nullable=False,
        quality_score=0.98
    )
    profile.add_column_profile(email_profile)
    
    return profile


@pytest.fixture
def check_with_results():
    """Data quality check with execution history."""
    check = DataQualityCheck(
        name="Completeness Check",
        description="Check for null values",
        check_type=CheckType.COMPLETENESS,
        dataset_name="orders",
        column_name="customer_id",
        threshold=0.99,
        created_by="engineer"
    )
    
    # Execute the check to create results
    result = check.execute()
    return check


@pytest.fixture
def rule_with_multiple_conditions():
    """Data quality rule with multiple conditions."""
    rule = DataQualityRule(
        name="Customer Validation",
        description="Validate customer data integrity",
        rule_type=RuleType.BUSINESS,
        severity=RuleSeverity.ERROR,
        dataset_name="customers",
        logical_operator="AND",
        created_by="engineer"
    )
    
    # Add multiple conditions
    age_condition = RuleCondition(
        column_name="age",
        operator=RuleOperator.GREATER_EQUAL,
        value=18,
        description="Must be 18 or older"
    )
    
    email_condition = RuleCondition(
        column_name="email",
        operator=RuleOperator.IS_NOT_NULL,
        description="Email is required"
    )
    
    rule.add_condition(age_condition)
    rule.add_condition(email_condition)
    
    return rule


@pytest.fixture
def completed_data_profile():
    """Completed data profile with full statistics."""
    profile = DataProfile(
        dataset_name="sales_data",
        table_name="transactions",
        total_rows=10000,
        total_columns=8,
        created_by="analyst"
    )
    
    # Start and complete profiling
    profile.start_profiling()
    
    # Add column profiles
    amount_stats = ProfileStatistics(
        total_count=10000,
        null_count=0,
        distinct_count=9500,
        min_value=0.01,
        max_value=9999.99,
        mean=250.75,
        median=195.50,
        std_dev=180.25
    )
    
    amount_profile = ColumnProfile(
        column_name="amount",
        data_type=DataType.DECIMAL,
        position=1,
        is_nullable=False,
        statistics=amount_stats,
        quality_score=0.98
    )
    
    profile.add_column_profile(amount_profile)
    profile.complete_profiling()
    
    return profile