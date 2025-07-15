"""Tests for the validation engine."""

import pytest
import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta

from ..application.services.validation_engine import ValidationEngine
from ..domain.entities.quality_rule import (
    QualityRule, RuleType, LogicType, ValidationLogic, QualityThreshold,
    Severity, UserId, DatasetId, RuleId, ValidationStatus
)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', None, 'Bob Johnson', ''],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com', None],
        'age': [30, 25, 35, None, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'created_at': [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2020, 1, 1),  # Old record
            datetime(2023, 1, 5)
        ]
    })


@pytest.fixture
def validation_engine():
    """Create validation engine instance."""
    return ValidationEngine()


@pytest.fixture
def completeness_rule():
    """Create completeness validation rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="Name Completeness",
        rule_type=RuleType.COMPLETENESS,
        target_columns=['name'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.PYTHON,
            expression="df['name'].notna() & (df['name'].str.strip() != '')",
            parameters={},
            error_message_template="Missing or empty name in row {row}"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=0.8,
            warning_threshold=0.7,
            critical_threshold=0.5
        ),
        severity=Severity.HIGH,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )


@pytest.fixture
def uniqueness_rule():
    """Create uniqueness validation rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="ID Uniqueness",
        rule_type=RuleType.UNIQUENESS,
        target_columns=['id'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.PYTHON,
            expression="df['id'].duplicated()",
            parameters={},
            error_message_template="Duplicate ID found: {value}"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=1.0,
            warning_threshold=0.95,
            critical_threshold=0.9
        ),
        severity=Severity.CRITICAL,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )


@pytest.fixture
def validity_rule():
    """Create validity validation rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="Email Format",
        rule_type=RuleType.VALIDITY,
        target_columns=['email'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.REGEX,
            expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            parameters={},
            error_message_template="Invalid email format: {value}"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=0.9,
            warning_threshold=0.8,
            critical_threshold=0.7
        ),
        severity=Severity.MEDIUM,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )


@pytest.fixture
def range_rule():
    """Create range validation rule."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="Age Range",
        rule_type=RuleType.VALIDITY,
        target_columns=['age'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.RANGE,
            expression="",
            parameters={'min_value': 18, 'max_value': 65},
            error_message_template="Age {value} is outside valid range"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=0.95,
            warning_threshold=0.9,
            critical_threshold=0.8
        ),
        severity=Severity.MEDIUM,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )


class TestValidationEngine:
    """Test cases for ValidationEngine."""
    
    def test_validate_empty_dataset(self, validation_engine, completeness_rule):
        """Test validation with empty dataset."""
        df = pd.DataFrame()
        dataset_id = DatasetId(value=uuid4())
        
        results = validation_engine.validate_dataset([completeness_rule], df, dataset_id)
        
        assert len(results) == 1
        assert results[0].total_records == 0
        assert results[0].pass_rate == 0.0
        assert results[0].status == ValidationStatus.FAILED
    
    def test_validate_completeness_rule(self, validation_engine, completeness_rule, sample_dataframe):
        """Test completeness validation rule."""
        dataset_id = DatasetId(value=uuid4())
        
        results = validation_engine.validate_dataset([completeness_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.rule_id == completeness_rule.rule_id
        assert result.total_records == 5
        assert result.records_failed == 2  # None and empty string
        assert result.records_passed == 3
        assert result.pass_rate == 0.6
        assert result.status == ValidationStatus.FAILED  # Below 0.8 threshold
        assert len(result.validation_errors) > 0
    
    def test_validate_uniqueness_rule(self, validation_engine, uniqueness_rule, sample_dataframe):
        """Test uniqueness validation rule."""
        dataset_id = DatasetId(value=uuid4())
        
        results = validation_engine.validate_dataset([uniqueness_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.rule_id == uniqueness_rule.rule_id
        assert result.total_records == 5
        assert result.records_failed == 0  # All IDs are unique
        assert result.records_passed == 5
        assert result.pass_rate == 1.0
        assert result.status == ValidationStatus.PASSED
    
    def test_validate_validity_rule(self, validation_engine, validity_rule, sample_dataframe):
        """Test validity validation rule."""
        dataset_id = DatasetId(value=uuid4())
        
        results = validation_engine.validate_dataset([validity_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.rule_id == validity_rule.rule_id
        assert result.total_records == 5
        assert result.records_failed == 2  # 'invalid-email' and None
        assert result.records_passed == 3
        assert result.pass_rate == 0.6
        assert result.status == ValidationStatus.FAILED  # Below 0.9 threshold
    
    def test_validate_range_rule(self, validation_engine, range_rule, sample_dataframe):
        """Test range validation rule."""
        dataset_id = DatasetId(value=uuid4())
        
        results = validation_engine.validate_dataset([range_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.rule_id == range_rule.rule_id
        assert result.total_records == 5
        # All ages (30, 25, 35, 45) are within 18-65 range, None is ignored
        assert result.records_failed == 0
        assert result.pass_rate >= 0.95
        assert result.status == ValidationStatus.PASSED
    
    def test_validate_multiple_rules(self, validation_engine, completeness_rule, uniqueness_rule, sample_dataframe):
        """Test validation with multiple rules."""
        dataset_id = DatasetId(value=uuid4())
        rules = [completeness_rule, uniqueness_rule]
        
        results = validation_engine.validate_dataset(rules, sample_dataframe, dataset_id)
        
        assert len(results) == 2
        
        # Check that both rules were executed
        rule_ids = [r.rule_id for r in results]
        assert completeness_rule.rule_id in rule_ids
        assert uniqueness_rule.rule_id in rule_ids
    
    def test_validate_disabled_rule(self, validation_engine, sample_dataframe):
        """Test that disabled rules are not executed."""
        disabled_rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Disabled Rule",
            rule_type=RuleType.COMPLETENESS,
            target_columns=['name'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="df['name'].notna()",
                parameters={},
                error_message_template="Missing name"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.MEDIUM,
            created_by=UserId(value=uuid4()),
            is_enabled=False  # Disabled
        )
        
        dataset_id = DatasetId(value=uuid4())
        results = validation_engine.validate_dataset([disabled_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 0  # No results for disabled rules
    
    def test_validate_single_record(self, validation_engine, completeness_rule):
        """Test single record validation."""
        # Valid record
        valid_record = {'name': 'John Doe', 'email': 'john@example.com'}
        result = validation_engine.validate_single_record([completeness_rule], valid_record)
        
        assert len(result) == 1
        assert result[str(completeness_rule.rule_id.value)] == True
        
        # Invalid record
        invalid_record = {'name': '', 'email': 'john@example.com'}
        result = validation_engine.validate_single_record([completeness_rule], invalid_record)
        
        assert len(result) == 1
        assert result[str(completeness_rule.rule_id.value)] == False
    
    def test_validation_summary(self, validation_engine, completeness_rule, uniqueness_rule, sample_dataframe):
        """Test validation summary generation."""
        dataset_id = DatasetId(value=uuid4())
        rules = [completeness_rule, uniqueness_rule]
        
        results = validation_engine.validate_dataset(rules, sample_dataframe, dataset_id)
        summary = validation_engine.get_validation_summary(results)
        
        assert summary['total_rules'] == 2
        assert summary['passed_rules'] >= 0
        assert summary['failed_rules'] >= 0
        assert summary['error_rules'] == 0
        assert summary['total_records_validated'] > 0
        assert 0 <= summary['overall_pass_rate'] <= 1
        assert 'rule_breakdown' in summary
    
    def test_parallel_execution(self, validation_engine, sample_dataframe):
        """Test parallel rule execution."""
        # Create multiple rules
        rules = []
        for i in range(5):
            rule = QualityRule(
                rule_id=RuleId(value=uuid4()),
                rule_name=f"Rule {i}",
                rule_type=RuleType.COMPLETENESS,
                target_columns=['name'],
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df['name'].notna()",
                    parameters={},
                    error_message_template="Missing name"
                ),
                thresholds=QualityThreshold(
                    pass_rate_threshold=0.8,
                    warning_threshold=0.7,
                    critical_threshold=0.5
                ),
                severity=Severity.MEDIUM,
                created_by=UserId(value=uuid4()),
                is_enabled=True
            )
            rules.append(rule)
        
        dataset_id = DatasetId(value=uuid4())
        results = validation_engine.validate_dataset(rules, sample_dataframe, dataset_id)
        
        assert len(results) == 5
        # All rules should have been executed successfully
        assert all(r.status != ValidationStatus.ERROR for r in results)
    
    def test_error_handling(self, validation_engine, sample_dataframe):
        """Test error handling in validation."""
        # Create rule with invalid expression
        invalid_rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Invalid Rule",
            rule_type=RuleType.COMPLETENESS,
            target_columns=['nonexistent_column'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="df['nonexistent_column'].invalid_method()",
                parameters={},
                error_message_template="Error occurred"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.MEDIUM,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        dataset_id = DatasetId(value=uuid4())
        results = validation_engine.validate_dataset([invalid_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.ERROR
        assert len(results[0].validation_errors) > 0
    
    def test_custom_rule_execution(self, validation_engine, sample_dataframe):
        """Test custom rule execution."""
        custom_rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Custom Business Rule",
            rule_type=RuleType.CUSTOM,
            target_columns=['salary', 'age'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="(df['salary'] >= 40000) & (df['age'] >= 21)",
                parameters={'min_salary': 40000, 'min_age': 21},
                error_message_template="Business rule violation"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.HIGH,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        dataset_id = DatasetId(value=uuid4())
        results = validation_engine.validate_dataset([custom_rule], sample_dataframe, dataset_id)
        
        assert len(results) == 1
        result = results[0]
        assert result.rule_id == custom_rule.rule_id
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED]


class TestRuleExecutors:
    """Test individual rule executors."""
    
    def test_completeness_executor(self, validation_engine, sample_dataframe):
        """Test completeness rule executor specifically."""
        rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Test Completeness",
            rule_type=RuleType.COMPLETENESS,
            target_columns=['name', 'email'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="",
                parameters={},
                error_message_template="Missing value"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.MEDIUM,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        executor = validation_engine._rule_executors[RuleType.COMPLETENESS]
        result = executor.execute(rule, sample_dataframe, DatasetId(value=uuid4()), None)
        
        assert result.rule_id == rule.rule_id
        assert result.total_records == 5
        assert result.records_failed > 0  # Should find missing values
    
    def test_validity_executor_with_regex(self, validation_engine, sample_dataframe):
        """Test validity rule executor with regex logic."""
        rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Email Regex Validation",
            rule_type=RuleType.VALIDITY,
            target_columns=['email'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.REGEX,
                expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                parameters={},
                error_message_template="Invalid email format"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.MEDIUM,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        executor = validation_engine._rule_executors[RuleType.VALIDITY]
        result = executor.execute(rule, sample_dataframe, DatasetId(value=uuid4()), None)
        
        assert result.rule_id == rule.rule_id
        assert result.total_records == 5
        # Should find invalid emails
        assert result.records_failed >= 1
    
    def test_timeliness_executor(self, validation_engine, sample_dataframe):
        """Test timeliness rule executor."""
        rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name="Data Freshness",
            rule_type=RuleType.TIMELINESS,
            target_columns=['created_at'],
            validation_logic=ValidationLogic(
                logic_type=LogicType.PYTHON,
                expression="",
                parameters={'max_age_hours': 8760},  # 1 year
                error_message_template="Data is too old"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=0.8,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            severity=Severity.MEDIUM,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        executor = validation_engine._rule_executors[RuleType.TIMELINESS]
        result = executor.execute(rule, sample_dataframe, DatasetId(value=uuid4()), None)
        
        assert result.rule_id == rule.rule_id
        assert result.total_records == 5
        # Should find old records (2020 record)
        assert result.records_failed >= 1