"""Comprehensive unit tests for RuleEvaluator."""

import pytest
import pandas as pd
import numpy as np
import re
from unittest.mock import Mock, patch
from datetime import datetime, date
from decimal import Decimal

from src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator


@pytest.mark.unit
class TestRuleEvaluator:
    """Test suite for RuleEvaluator."""

    @pytest.fixture
    def rule_evaluator(self):
        """Create RuleEvaluator instance for testing."""
        return RuleEvaluator()

    @pytest.fixture
    def sample_record(self):
        """Create sample record for testing."""
        return {
            'id': 1,
            'name': 'John Doe',
            'age': 30,
            'email': 'john.doe@example.com',
            'salary': 50000.0,
            'department': 'Engineering',
            'join_date': '2020-01-15',
            'is_active': True,
            'rating': 4.5,
            'notes': 'Good performance'
        }

    def test_evaluate_not_null_rule_success(self, rule_evaluator, sample_record):
        """Test NOT_NULL rule evaluation - success case."""
        # Arrange
        rule = DataQualityRule(
            name="name_not_null",
            description="Name should not be null",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="name",
            operator=RuleOperator.NOT_EQUALS,
            value="null"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_not_null_rule_failure(self, rule_evaluator):
        """Test NOT_NULL rule evaluation - failure case."""
        # Arrange
        record_with_null = {
            'id': 1,
            'name': None,
            'age': 30
        }
        
        rule = DataQualityRule(
            name="name_not_null",
            description="Name should not be null",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="name",
            operator=RuleOperator.NOT_EQUALS,
            value="null"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record_with_null, rule)
        
        # Assert
        assert result is False

    def test_evaluate_unique_rule_success(self, rule_evaluator, sample_record):
        """Test UNIQUE rule evaluation - success case."""
        # Arrange
        rule = DataQualityRule(
            name="id_unique",
            description="ID should be unique",
            rule_type=RuleType.UNIQUE,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="id",
            operator=RuleOperator.UNIQUE,
            value=""
        )
        rule.add_condition(condition)
        
        # Mock unique check (assume it passes)
        with patch.object(rule_evaluator, '_check_unique_constraint', return_value=True):
            # Act
            result = rule_evaluator.evaluate_record(sample_record, rule)
            
            # Assert
            assert result is True

    def test_evaluate_range_rule_success(self, rule_evaluator, sample_record):
        """Test RANGE rule evaluation - success case."""
        # Arrange
        rule = DataQualityRule(
            name="age_range",
            description="Age should be between 18 and 65",
            rule_type=RuleType.RANGE,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.BETWEEN,
            value="18,65"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_range_rule_failure(self, rule_evaluator):
        """Test RANGE rule evaluation - failure case."""
        # Arrange
        record_invalid_age = {
            'id': 1,
            'name': 'John Doe',
            'age': 150  # Invalid age
        }
        
        rule = DataQualityRule(
            name="age_range",
            description="Age should be between 18 and 65",
            rule_type=RuleType.RANGE,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.BETWEEN,
            value="18,65"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record_invalid_age, rule)
        
        # Assert
        assert result is False

    def test_evaluate_pattern_rule_success(self, rule_evaluator, sample_record):
        """Test PATTERN rule evaluation - success case."""
        # Arrange
        rule = DataQualityRule(
            name="email_pattern",
            description="Email should match pattern",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="email",
            operator=RuleOperator.PATTERN,
            value=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_pattern_rule_failure(self, rule_evaluator):
        """Test PATTERN rule evaluation - failure case."""
        # Arrange
        record_invalid_email = {
            'id': 1,
            'name': 'John Doe',
            'email': 'invalid-email'  # Invalid email format
        }
        
        rule = DataQualityRule(
            name="email_pattern",
            description="Email should match pattern",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="email",
            operator=RuleOperator.PATTERN,
            value=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record_invalid_email, rule)
        
        # Assert
        assert result is False

    def test_evaluate_custom_rule_with_and_logic(self, rule_evaluator, sample_record):
        """Test CUSTOM rule evaluation with AND logic."""
        # Arrange
        rule = DataQualityRule(
            name="complex_validation",
            description="Complex validation with AND logic",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset",
            logical_operator="AND"
        )
        
        condition1 = RuleCondition(
            column_name="age",
            operator=RuleOperator.GREATER_THAN,
            value="18"
        )
        condition2 = RuleCondition(
            column_name="salary",
            operator=RuleOperator.GREATER_THAN,
            value="0"
        )
        
        rule.add_condition(condition1)
        rule.add_condition(condition2)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_custom_rule_with_or_logic(self, rule_evaluator):
        """Test CUSTOM rule evaluation with OR logic."""
        # Arrange
        record = {
            'id': 1,
            'department': 'Engineering',
            'level': 'Junior'
        }
        
        rule = DataQualityRule(
            name="department_or_level",
            description="Either department should be Engineering OR level should be Senior",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset",
            logical_operator="OR"
        )
        
        condition1 = RuleCondition(
            column_name="department",
            operator=RuleOperator.EQUALS,
            value="Engineering"
        )
        condition2 = RuleCondition(
            column_name="level",
            operator=RuleOperator.EQUALS,
            value="Senior"
        )
        
        rule.add_condition(condition1)
        rule.add_condition(condition2)
        
        # Act
        result = rule_evaluator.evaluate_record(record, rule)
        
        # Assert
        assert result is True  # Passes because department is Engineering

    def test_evaluate_equals_operator(self, rule_evaluator, sample_record):
        """Test EQUALS operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="department_check",
            description="Department should be Engineering",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="department",
            operator=RuleOperator.EQUALS,
            value="Engineering"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_not_equals_operator(self, rule_evaluator, sample_record):
        """Test NOT_EQUALS operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="department_not_hr",
            description="Department should not be HR",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="department",
            operator=RuleOperator.NOT_EQUALS,
            value="HR"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_greater_than_operator(self, rule_evaluator, sample_record):
        """Test GREATER_THAN operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="salary_minimum",
            description="Salary should be greater than 40000",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="salary",
            operator=RuleOperator.GREATER_THAN,
            value="40000"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_less_than_operator(self, rule_evaluator, sample_record):
        """Test LESS_THAN operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="age_maximum",
            description="Age should be less than 65",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.LESS_THAN,
            value="65"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_greater_than_or_equal_operator(self, rule_evaluator, sample_record):
        """Test GREATER_THAN_OR_EQUAL operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="age_minimum",
            description="Age should be at least 18",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.GREATER_THAN_OR_EQUAL,
            value="18"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_less_than_or_equal_operator(self, rule_evaluator, sample_record):
        """Test LESS_THAN_OR_EQUAL operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="rating_maximum",
            description="Rating should be at most 5.0",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="rating",
            operator=RuleOperator.LESS_THAN_OR_EQUAL,
            value="5.0"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_in_operator(self, rule_evaluator, sample_record):
        """Test IN operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="department_allowed",
            description="Department should be in allowed list",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="department",
            operator=RuleOperator.IN,
            value="Engineering,Sales,Marketing,HR"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_not_in_operator(self, rule_evaluator, sample_record):
        """Test NOT_IN operator evaluation."""
        # Arrange
        rule = DataQualityRule(
            name="department_not_restricted",
            description="Department should not be in restricted list",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="department",
            operator=RuleOperator.NOT_IN,
            value="Finance,Legal"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(sample_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_missing_column(self, rule_evaluator):
        """Test evaluation when column is missing from record."""
        # Arrange
        incomplete_record = {
            'id': 1,
            'name': 'John Doe'
            # Missing age column
        }
        
        rule = DataQualityRule(
            name="age_check",
            description="Age should be greater than 18",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.GREATER_THAN,
            value="18"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(incomplete_record, rule)
        
        # Assert
        assert result is False  # Should fail when column is missing

    def test_evaluate_with_null_values(self, rule_evaluator):
        """Test evaluation with various null value representations."""
        # Arrange
        record_with_nulls = {
            'id': None,
            'name': '',
            'age': np.nan,
            'salary': 'NULL',
            'department': 'N/A'
        }
        
        rule = DataQualityRule(
            name="null_check",
            description="ID should not be null",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="id",
            operator=RuleOperator.NOT_EQUALS,
            value="null"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record_with_nulls, rule)
        
        # Assert
        assert result is False

    def test_evaluate_with_different_data_types(self, rule_evaluator):
        """Test evaluation with different data types."""
        # Arrange
        record_mixed_types = {
            'int_field': 42,
            'float_field': 3.14,
            'string_field': 'test',
            'bool_field': True,
            'date_field': datetime(2023, 1, 1),
            'decimal_field': Decimal('123.45')
        }
        
        rule = DataQualityRule(
            name="type_check",
            description="Integer field should be greater than 0",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="int_field",
            operator=RuleOperator.GREATER_THAN,
            value="0"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record_mixed_types, rule)
        
        # Assert
        assert result is True

    def test_evaluate_complex_pattern(self, rule_evaluator):
        """Test evaluation with complex regex pattern."""
        # Arrange
        record = {
            'phone': '+1-555-123-4567'
        }
        
        rule = DataQualityRule(
            name="phone_pattern",
            description="Phone should match international format",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.WARNING,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="phone",
            operator=RuleOperator.PATTERN,
            value=r"^\+\d{1,3}-\d{3}-\d{3}-\d{4}$"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_case_insensitive_comparison(self, rule_evaluator):
        """Test case-insensitive string comparison."""
        # Arrange
        record = {
            'department': 'engineering'  # lowercase
        }
        
        rule = DataQualityRule(
            name="department_case_insensitive",
            description="Department should be Engineering (case insensitive)",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="department",
            operator=RuleOperator.EQUALS,
            value="Engineering",
            case_sensitive=False
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_date_comparison(self, rule_evaluator):
        """Test date comparison evaluation."""
        # Arrange
        record = {
            'join_date': '2023-06-15'
        }
        
        rule = DataQualityRule(
            name="join_date_recent",
            description="Join date should be after 2020-01-01",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="join_date",
            operator=RuleOperator.GREATER_THAN,
            value="2020-01-01"
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_with_exception_handling(self, rule_evaluator, sample_record):
        """Test evaluation with exception handling."""
        # Arrange
        rule = DataQualityRule(
            name="invalid_pattern",
            description="Rule with invalid regex pattern",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="name",
            operator=RuleOperator.PATTERN,
            value="[invalid_regex"  # Invalid regex
        )
        rule.add_condition(condition)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            rule_evaluator.evaluate_record(sample_record, rule)

    def test_evaluate_empty_rule_conditions(self, rule_evaluator, sample_record):
        """Test evaluation with empty rule conditions."""
        # Arrange
        rule = DataQualityRule(
            name="empty_rule",
            description="Rule with no conditions",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        # No conditions added
        
        # Act & Assert
        with pytest.raises(ValueError, match="Rule must have at least one condition"):
            rule_evaluator.evaluate_record(sample_record, rule)

    @pytest.mark.parametrize("operator,value,record_value,expected", [
        (RuleOperator.EQUALS, "test", "test", True),
        (RuleOperator.EQUALS, "test", "TEST", False),
        (RuleOperator.NOT_EQUALS, "test", "other", True),
        (RuleOperator.GREATER_THAN, "10", "15", True),
        (RuleOperator.GREATER_THAN, "10", "5", False),
        (RuleOperator.LESS_THAN, "20", "15", True),
        (RuleOperator.LESS_THAN, "20", "25", False),
        (RuleOperator.BETWEEN, "10,20", "15", True),
        (RuleOperator.BETWEEN, "10,20", "25", False),
        (RuleOperator.IN, "a,b,c", "b", True),
        (RuleOperator.IN, "a,b,c", "d", False),
        (RuleOperator.NOT_IN, "a,b,c", "d", True),
        (RuleOperator.NOT_IN, "a,b,c", "b", False),
    ])
    def test_evaluate_operators_parametrized(self, rule_evaluator, operator, value, record_value, expected):
        """Test various operators with parametrized values."""
        # Arrange
        record = {'field': record_value}
        rule = DataQualityRule(
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="field",
            operator=operator,
            value=value
        )
        rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(record, rule)
        
        # Assert
        assert result == expected

    def test_evaluate_performance_with_large_record(self, rule_evaluator, performance_monitor):
        """Test evaluation performance with large record."""
        # Arrange
        large_record = {}
        for i in range(1000):
            large_record[f'field_{i}'] = f'value_{i}'
        
        rule = DataQualityRule(
            name="performance_test",
            description="Performance test rule",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="field_500",
            operator=RuleOperator.EQUALS,
            value="value_500"
        )
        rule.add_condition(condition)
        
        # Act
        performance_monitor.start()
        result = rule_evaluator.evaluate_record(large_record, rule)
        metrics = performance_monitor.stop()
        
        # Assert
        assert result is True
        assert metrics['duration_seconds'] < 1.0  # Should be fast

    def test_evaluate_concurrent_evaluations(self, rule_evaluator, sample_record):
        """Test concurrent rule evaluations."""
        import threading
        
        # Arrange
        rule = DataQualityRule(
            name="concurrent_test",
            description="Concurrent test rule",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.INFO,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.GREATER_THAN,
            value="18"
        )
        rule.add_condition(condition)
        
        results = []
        errors = []
        
        def evaluate_thread():
            try:
                result = rule_evaluator.evaluate_record(sample_record, rule)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=evaluate_thread)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 10
        assert len(errors) == 0
        assert all(result is True for result in results)


@pytest.mark.integration
class TestRuleEvaluatorIntegration:
    """Integration tests for RuleEvaluator with real data."""

    def test_evaluate_with_pandas_dataframe(self, rule_evaluator, sample_csv_data):
        """Test evaluation with pandas DataFrame records."""
        # Arrange
        rule = DataQualityRule(
            name="salary_validation",
            description="Salary should be positive",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        condition = RuleCondition(
            column_name="salary",
            operator=RuleOperator.GREATER_THAN,
            value="0"
        )
        rule.add_condition(condition)
        
        # Act
        results = []
        for _, row in sample_csv_data.iterrows():
            record = row.to_dict()
            result = rule_evaluator.evaluate_record(record, rule)
            results.append(result)
        
        # Assert
        assert len(results) == len(sample_csv_data)
        assert all(isinstance(result, bool) for result in results)

    def test_evaluate_complex_business_rules(self, rule_evaluator):
        """Test evaluation with complex business rules."""
        # Arrange
        employee_record = {
            'employee_id': 'EMP001',
            'name': 'John Doe',
            'age': 28,
            'department': 'Engineering',
            'level': 'Senior',
            'salary': 85000,
            'years_experience': 6,
            'performance_rating': 4.2,
            'has_degree': True,
            'start_date': '2018-03-15'
        }
        
        # Complex rule: Senior engineers should have >5 years experience AND salary >80k AND degree
        rule = DataQualityRule(
            name="senior_engineer_validation",
            description="Senior engineers must meet multiple criteria",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="employees",
            logical_operator="AND"
        )
        
        conditions = [
            RuleCondition(column_name="level", operator=RuleOperator.EQUALS, value="Senior"),
            RuleCondition(column_name="years_experience", operator=RuleOperator.GREATER_THAN, value="5"),
            RuleCondition(column_name="salary", operator=RuleOperator.GREATER_THAN, value="80000"),
            RuleCondition(column_name="has_degree", operator=RuleOperator.EQUALS, value="True")
        ]
        
        for condition in conditions:
            rule.add_condition(condition)
        
        # Act
        result = rule_evaluator.evaluate_record(employee_record, rule)
        
        # Assert
        assert result is True

    def test_evaluate_data_quality_scenarios(self, rule_evaluator, corrupted_csv_data):
        """Test evaluation with real data quality scenarios."""
        # Arrange
        rules = [
            # Age validation rule
            DataQualityRule(
                name="age_range_check",
                description="Age should be between 0 and 120",
                rule_type=RuleType.RANGE,
                severity=RuleSeverity.ERROR,
                dataset_name="test_dataset"
            ),
            # Email pattern rule
            DataQualityRule(
                name="email_format_check",
                description="Email should be valid format",
                rule_type=RuleType.PATTERN,
                severity=RuleSeverity.WARNING,
                dataset_name="test_dataset"
            ),
            # Department allowlist rule
            DataQualityRule(
                name="department_allowlist",
                description="Department should be in allowed list",
                rule_type=RuleType.CUSTOM,
                severity=RuleSeverity.ERROR,
                dataset_name="test_dataset"
            )
        ]
        
        # Add conditions to rules
        rules[0].add_condition(RuleCondition("age", RuleOperator.BETWEEN, "0,120"))
        rules[1].add_condition(RuleCondition("email", RuleOperator.PATTERN, r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"))
        rules[2].add_condition(RuleCondition("department", RuleOperator.IN, "Engineering,Sales,Marketing,HR"))
        
        # Act
        validation_results = {}
        for rule in rules:
            rule_results = []
            for _, row in corrupted_csv_data.iterrows():
                record = row.to_dict()
                try:
                    result = rule_evaluator.evaluate_record(record, rule)
                    rule_results.append(result)
                except Exception as e:
                    rule_results.append(False)  # Treat exceptions as failures
            validation_results[rule.name] = rule_results
        
        # Assert
        assert len(validation_results) == 3
        
        # Age rule should catch invalid ages
        age_results = validation_results["age_range_check"]
        assert not all(age_results)  # Should have some failures
        
        # Email rule should catch invalid emails
        email_results = validation_results["email_format_check"]
        assert not all(email_results)  # Should have some failures
        
        # Department rule should catch invalid departments
        dept_results = validation_results["department_allowlist"]
        assert not all(dept_results)  # Should have some failures