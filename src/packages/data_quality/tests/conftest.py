import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import Mock
from uuid import uuid4
from datetime import datetime, timedelta

from data_quality.domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, Severity,
    RuleStatus, ValidationStatus, LogicType, QualityCategory,
    ValidationLogic, RuleSchedule, ValidationError, ValidationResult,
    QualityThreshold, RuleMetadata
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for quality testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' if i % 20 != 0 else None for i in range(1, 1001)],  # 5% missing
        'email': [
            f'user{i}@example.com' if i % 50 != 0 else 'invalid-email'  # 2% invalid
            for i in range(1, 1001)
        ],
        'age': [
            np.random.randint(18, 80) if i % 100 != 0 else -1  # 1% invalid ages
            for i in range(1, 1001)
        ],
        'salary': np.random.normal(50000, 15000, 1000),
        'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', None], 1000),
        'created_at': pd.date_range('2020-01-01', periods=1000, freq='H')
    })


@pytest.fixture
def invalid_dataframe():
    """Create a DataFrame with quality issues for testing."""
    return pd.DataFrame({
        'id': [1, 2, 2, 4, None, 6, 7, 8, 9, 10],  # Duplicate and null IDs
        'email': [
            'valid@example.com',
            'invalid-email',
            'another@valid.com',
            '',
            None,
            'no-at-sign.com',
            'valid2@example.com',
            'spaces in email@example.com',
            'valid3@example.com',
            'valid4@example.com'
        ],
        'age': [-5, 25, 30, 150, None, 35, 40, 45, 50, 200],  # Invalid ages
        'salary': [-1000, 25000, 30000, 1000000, None, 35000, 40000, 45000, 50000, 55000]
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def rule_id():
    """Create a rule ID."""
    return RuleId(value=uuid4())


@pytest.fixture
def dataset_id():
    """Create a dataset ID."""
    return DatasetId(value=uuid4())


@pytest.fixture
def user_id():
    """Create a user ID."""
    return UserId(value=uuid4())


@pytest.fixture
def completeness_validation_logic():
    """Create completeness validation logic."""
    return ValidationLogic(
        logic_type=LogicType.SQL,
        expression="COUNT(*) WHERE column_name IS NOT NULL",
        parameters={"column_name": "name"},
        error_message_template="Missing values found in {column_name}",
        success_criteria="pass_rate >= 0.95"
    )


@pytest.fixture
def email_validation_logic():
    """Create email validation logic."""
    return ValidationLogic(
        logic_type=LogicType.REGEX,
        expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        parameters={"column_name": "email"},
        error_message_template="Invalid email format in {column_name}",
        success_criteria="pass_rate >= 0.98"
    )


@pytest.fixture
def range_validation_logic():
    """Create range validation logic."""
    return ValidationLogic(
        logic_type=LogicType.RANGE,
        expression="BETWEEN 18 AND 80",
        parameters={"column_name": "age", "min_value": 18, "max_value": 80},
        error_message_template="Age value out of valid range (18-80)",
        success_criteria="pass_rate >= 0.99"
    )


@pytest.fixture
def quality_threshold():
    """Create quality threshold."""
    return QualityThreshold(
        pass_rate_threshold=0.95,
        warning_threshold=0.90,
        critical_threshold=0.80
    )


@pytest.fixture
def daily_schedule():
    """Create daily schedule."""
    return RuleSchedule(
        enabled=True,
        frequency="daily",
        next_execution=datetime.utcnow() + timedelta(days=1)
    )


@pytest.fixture
def rule_metadata():
    """Create rule metadata."""
    return RuleMetadata(
        description="Validates that name field is not null",
        business_justification="Name is required for all user records",
        data_owner="Data Team",
        business_glossary_terms=["customer", "identity"],
        related_regulations=["GDPR", "CCPA"],
        documentation_url="https://wiki.company.com/data-quality/name-validation"
    )


@pytest.fixture
def completeness_rule(
    rule_id,
    dataset_id,
    user_id,
    completeness_validation_logic,
    quality_threshold,
    rule_metadata
):
    """Create a completeness quality rule."""
    return QualityRule(
        rule_id=rule_id,
        rule_name="name_completeness_check",
        rule_type=RuleType.COMPLETENESS,
        category=QualityCategory.DATA_INTEGRITY,
        severity=Severity.HIGH,
        status=RuleStatus.DRAFT,
        validation_logic=completeness_validation_logic,
        target_datasets=[dataset_id],
        target_columns=["name"],
        thresholds=quality_threshold,
        metadata=rule_metadata,
        created_by=user_id
    )


@pytest.fixture
def email_validity_rule(
    user_id,
    email_validation_logic,
    quality_threshold,
    rule_metadata
):
    """Create an email validity rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="email_format_validation",
        rule_type=RuleType.VALIDITY,
        category=QualityCategory.FORMAT_VALIDATION,
        severity=Severity.MEDIUM,
        status=RuleStatus.ACTIVE,
        validation_logic=email_validation_logic,
        target_columns=["email"],
        thresholds=quality_threshold,
        metadata=rule_metadata,
        created_by=user_id
    )


@pytest.fixture
def age_range_rule(
    user_id,
    range_validation_logic,
    quality_threshold,
    daily_schedule,
    rule_metadata
):
    """Create an age range validation rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="age_range_validation",
        rule_type=RuleType.VALIDITY,
        category=QualityCategory.BUSINESS_RULES,
        severity=Severity.CRITICAL,
        status=RuleStatus.ACTIVE,
        validation_logic=range_validation_logic,
        target_columns=["age"],
        thresholds=quality_threshold,
        schedule=daily_schedule,
        metadata=rule_metadata,
        created_by=user_id
    )


@pytest.fixture
def validation_error():
    """Create a validation error."""
    return ValidationError(
        row_identifier="row_123",
        column_name="email",
        invalid_value="invalid-email",
        error_message="Invalid email format",
        error_code="FORMAT_ERROR"
    )


@pytest.fixture
def validation_result(rule_id, dataset_id, validation_error):
    """Create a validation result."""
    return ValidationResult(
        rule_id=rule_id,
        dataset_id=dataset_id,
        status=ValidationStatus.FAILED,
        total_records=1000,
        records_passed=980,
        records_failed=20,
        pass_rate=0.98,
        validation_errors=[validation_error],
        execution_time_seconds=2.5,
        memory_usage_mb=128.0
    )


@pytest.fixture
def passed_validation_result(rule_id, dataset_id):
    """Create a passed validation result."""
    return ValidationResult(
        rule_id=rule_id,
        dataset_id=dataset_id,
        status=ValidationStatus.PASSED,
        total_records=1000,
        records_passed=999,
        records_failed=1,
        pass_rate=0.999,
        validation_errors=[],
        execution_time_seconds=1.8,
        memory_usage_mb=96.0
    )


@pytest.fixture
def active_rule_with_results(completeness_rule, validation_result, user_id):
    """Create an active rule with validation results."""
    rule = completeness_rule
    rule.activate(approved_by=user_id)
    rule.add_validation_result(validation_result)
    return rule


@pytest.fixture
def mock_quality_rule_repository():
    """Create a mock quality rule repository."""
    return Mock()


@pytest.fixture
def mock_validation_result_repository():
    """Create a mock validation result repository."""
    return Mock()


@pytest.fixture
def mock_validation_engine():
    """Create a mock validation engine."""
    return Mock()


@pytest.fixture
def mock_cleansing_engine():
    """Create a mock cleansing engine."""
    return Mock()


@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    return Mock()


@pytest.fixture
def mock_monitoring_service():
    """Create a mock monitoring service."""
    return Mock()


@pytest.fixture
def quality_config():
    """Create quality configuration."""
    return {
        "validation": {
            "batch_size": 1000,
            "timeout_seconds": 300,
            "retry_attempts": 3
        },
        "thresholds": {
            "default_pass_rate": 0.95,
            "warning_threshold": 0.90,
            "critical_threshold": 0.80
        },
        "monitoring": {
            "enable_real_time": True,
            "alert_channels": ["email", "slack"],
            "dashboard_refresh_seconds": 30
        }
    }