"""Comprehensive unit tests for DataQualityCheckService."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, call
from uuid import uuid4
from datetime import datetime

from src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckType, CheckStatus, CheckSeverity, CheckResult
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity


@pytest.mark.unit
class TestDataQualityCheckService:
    """Test suite for DataQualityCheckService."""

    def test_run_check_success(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule, sample_csv_data):
        """Test successful check execution."""
        # Arrange
        check_id = sample_data_quality_check.id
        source_config = {"file_path": "test.csv"}
        
        # Setup mocks
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        result = data_quality_check_service.run_check(check_id, source_config)
        
        # Assert
        assert isinstance(result, DataQualityCheck)
        assert result.status == CheckStatus.COMPLETED
        assert result.result is not None
        assert isinstance(result.result, CheckResult)
        
        # Verify repository interactions
        data_quality_check_service.data_quality_check_repository.get_by_id.assert_called_once_with(check_id)
        data_quality_check_service.data_quality_check_repository.save.assert_called_once_with(result)

    def test_run_check_not_found(self, data_quality_check_service):
        """Test check execution when check is not found."""
        # Arrange
        check_id = uuid4()
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="DataQualityCheck with id .* not found"):
            data_quality_check_service.run_check(check_id, {})

    def test_run_check_rule_not_found(self, data_quality_check_service, sample_data_quality_check):
        """Test check execution when associated rule is not found."""
        # Arrange
        check_id = sample_data_quality_check.id
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="DataQualityRule with id .* not found"):
            data_quality_check_service.run_check(check_id, {})

    def test_run_check_data_loading_error(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule):
        """Test check execution when data loading fails."""
        # Arrange
        check_id = sample_data_quality_check.id
        source_config = {"file_path": "nonexistent.csv"}
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.side_effect = FileNotFoundError("File not found")
        
        # Act
        result = data_quality_check_service.run_check(check_id, source_config)
        
        # Assert
        assert result.status == CheckStatus.FAILED
        assert result.result is not None
        assert result.result.passed == False
        assert "File not found" in result.result.message

    def test_run_check_rule_evaluation_mixed_results(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule, sample_csv_data):
        """Test check execution with mixed rule evaluation results."""
        # Arrange
        check_id = sample_data_quality_check.id
        source_config = {"file_path": "test.csv"}
        
        # Setup mocks with mixed results (80% pass, 20% fail)
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        
        # Make 20% of evaluations fail
        evaluation_results = [True] * 80 + [False] * 20
        data_quality_check_service.rule_evaluator.evaluate_record.side_effect = evaluation_results
        
        # Act
        result = data_quality_check_service.run_check(check_id, source_config)
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.error_count == 20
        assert result.result.total_records == 100
        assert result.result.score == 0.8  # 80% pass rate

    def test_run_check_empty_dataset(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule):
        """Test check execution with empty dataset."""
        # Arrange
        check_id = sample_data_quality_check.id
        empty_data = pd.DataFrame()
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = empty_data
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.total_records == 0
        assert result.result.error_count == 0
        assert result.result.score == 1.0  # No errors in empty dataset

    def test_run_check_all_failures(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule, sample_csv_data):
        """Test check execution where all records fail validation."""
        # Arrange
        check_id = sample_data_quality_check.id
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = False  # All fail
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.error_count == len(sample_csv_data)
        assert result.result.score == 0.0
        assert result.result.passed == False

    def test_run_check_rule_evaluator_error(self, data_quality_check_service, sample_data_quality_check, sample_data_quality_rule, sample_csv_data):
        """Test check execution when rule evaluator raises an error."""
        # Arrange
        check_id = sample_data_quality_check.id
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.side_effect = ValueError("Invalid rule condition")
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.status == CheckStatus.FAILED
        assert "Invalid rule condition" in result.result.message

    def test_run_check_concurrent_execution(self, mock_data_quality_check_repository, mock_data_quality_rule_repository, 
                                          mock_pandas_csv_adapter, mock_rule_evaluator, sample_csv_data):
        """Test concurrent check execution."""
        import threading
        import time
        
        # Arrange
        service = DataQualityCheckService(
            mock_data_quality_check_repository,
            mock_data_quality_rule_repository, 
            mock_pandas_csv_adapter,
            mock_rule_evaluator
        )
        
        # Create multiple checks
        checks = []
        rules = []
        for i in range(3):
            check = DataQualityCheck(
                name=f"concurrent_check_{i}",
                description=f"Concurrent test check {i}",
                check_type=CheckType.VALIDATION,
                rule_id=uuid4(),
                dataset_name="test_dataset"
            )
            rule = DataQualityRule(
                name=f"concurrent_rule_{i}",
                description=f"Concurrent test rule {i}",
                rule_type=RuleType.NOT_NULL,
                severity=RuleSeverity.ERROR,
                dataset_name="test_dataset"
            )
            checks.append(check)
            rules.append(rule)
        
        # Setup mocks
        def get_check_by_id(check_id):
            return next((c for c in checks if c.id == check_id), None)
        
        def get_rule_by_id(rule_id):
            return next((r for r in rules if r.id == rule_id), None)
        
        mock_data_quality_check_repository.get_by_id.side_effect = get_check_by_id
        mock_data_quality_rule_repository.get_by_id.side_effect = get_rule_by_id
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        mock_rule_evaluator.evaluate_record.return_value = True
        
        results = []
        errors = []
        
        def run_check_thread(check):
            try:
                result = service.run_check(check.id, {})
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for check in checks:
            thread = threading.Thread(target=run_check_thread, args=(check,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 3
        assert len(errors) == 0
        assert all(result.status == CheckStatus.COMPLETED for result in results)

    def test_run_check_performance_large_dataset(self, data_quality_check_service, sample_data_quality_check, 
                                               sample_data_quality_rule, large_dataset, performance_monitor):
        """Test check execution performance with large dataset."""
        # Arrange
        check_id = sample_data_quality_check.id
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = large_dataset
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        performance_monitor.start()
        result = data_quality_check_service.run_check(check_id, {})
        metrics = performance_monitor.stop()
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.total_records == len(large_dataset)
        assert metrics['duration_seconds'] < 30  # Should complete within 30 seconds
        assert metrics['memory_delta_mb'] < 200  # Should not use excessive memory

    @pytest.mark.parametrize("rule_type,severity", [
        (RuleType.NOT_NULL, RuleSeverity.ERROR),
        (RuleType.UNIQUE, RuleSeverity.WARNING),
        (RuleType.RANGE, RuleSeverity.CRITICAL),
        (RuleType.PATTERN, RuleSeverity.INFO),
    ])
    def test_run_check_different_rule_types(self, data_quality_check_service, sample_csv_data, rule_type, severity):
        """Test check execution with different rule types and severities."""
        # Arrange
        rule = DataQualityRule(
            name=f"{rule_type.value}_test_rule",
            description=f"Test rule for {rule_type.value}",
            rule_type=rule_type,
            severity=severity,
            dataset_name="test_dataset"
        )
        
        check = DataQualityCheck(
            name=f"{rule_type.value}_test_check",
            description=f"Test check for {rule_type.value}",
            check_type=CheckType.VALIDATION,
            rule_id=rule.id,
            dataset_name="test_dataset"
        )
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        result = data_quality_check_service.run_check(check.id, {})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.passed == True

    def test_run_check_with_custom_source_config(self, data_quality_check_service, sample_data_quality_check, 
                                                sample_data_quality_rule, sample_csv_data):
        """Test check execution with custom source configuration."""
        # Arrange
        check_id = sample_data_quality_check.id
        custom_config = {
            "file_path": "custom.csv",
            "delimiter": ";",
            "encoding": "utf-8",
            "skiprows": 1
        }
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        result = data_quality_check_service.run_check(check_id, custom_config)
        
        # Assert
        data_quality_check_service.data_source_adapter.read_data.assert_called_once_with(custom_config)
        assert result.status == CheckStatus.COMPLETED

    def test_run_check_updates_timestamps(self, data_quality_check_service, sample_data_quality_check, 
                                        sample_data_quality_rule, sample_csv_data):
        """Test that check execution updates timestamps correctly."""
        # Arrange
        check_id = sample_data_quality_check.id
        original_updated_at = sample_data_quality_check.updated_at
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.updated_at > original_updated_at
        assert result.last_run_at is not None
        assert result.result.created_at is not None

    def test_run_check_preserves_check_metadata(self, data_quality_check_service, sample_data_quality_check, 
                                              sample_data_quality_rule, sample_csv_data):
        """Test that check execution preserves original check metadata."""
        # Arrange
        check_id = sample_data_quality_check.id
        original_name = sample_data_quality_check.name
        original_description = sample_data_quality_check.description
        original_created_at = sample_data_quality_check.created_at
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = sample_csv_data
        data_quality_check_service.rule_evaluator.evaluate_record.return_value = True
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.name == original_name
        assert result.description == original_description
        assert result.created_at == original_created_at
        assert result.id == check_id

    def test_run_check_calculates_score_correctly(self, data_quality_check_service, sample_data_quality_check, 
                                                sample_data_quality_rule):
        """Test that check execution calculates quality score correctly."""
        # Arrange
        check_id = sample_data_quality_check.id
        test_data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})  # 10 records
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = test_data
        
        # 7 pass, 3 fail
        evaluation_results = [True, True, True, True, True, True, True, False, False, False]
        data_quality_check_service.rule_evaluator.evaluate_record.side_effect = evaluation_results
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.result.total_records == 10
        assert result.result.error_count == 3
        assert result.result.score == 0.7  # 7/10 = 0.7
        assert result.result.passed == True  # Assuming > 0.5 is considered passed

    def test_run_check_handles_rule_evaluation_exceptions_gracefully(self, data_quality_check_service, 
                                                                   sample_data_quality_check, sample_data_quality_rule):
        """Test that check execution handles individual rule evaluation exceptions gracefully."""
        # Arrange
        check_id = sample_data_quality_check.id
        test_data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        data_quality_check_service.data_quality_check_repository.get_by_id.return_value = sample_data_quality_check
        data_quality_check_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_check_service.data_source_adapter.read_data.return_value = test_data
        
        # Mix of successful evaluations and exceptions
        def evaluation_side_effect(record):
            if record.get('col1', 0) == 3:
                raise ValueError("Evaluation error for record 3")
            return True
        
        data_quality_check_service.rule_evaluator.evaluate_record.side_effect = evaluation_side_effect
        
        # Act
        result = data_quality_check_service.run_check(check_id, {})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result.total_records == 5
        # The record that caused an exception should be counted as a failure
        assert result.result.error_count == 1
        assert result.result.score == 0.8  # 4/5 = 0.8


@pytest.mark.integration  
class TestDataQualityCheckServiceIntegration:
    """Integration tests for DataQualityCheckService."""

    def test_run_check_end_to_end_with_real_data(self, mock_data_quality_check_repository, 
                                                mock_data_quality_rule_repository, temp_csv_file):
        """Test end-to-end check execution with real CSV data."""
        from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter
        from src.data_quality.application.services.rule_evaluator import RuleEvaluator
        
        # Arrange
        adapter = PandasCSVAdapter()
        evaluator = RuleEvaluator()
        service = DataQualityCheckService(
            mock_data_quality_check_repository,
            mock_data_quality_rule_repository,
            adapter,
            evaluator
        )
        
        # Create real rule for age validation
        rule = create_test_rule("age_validation", RuleType.RANGE, "age", value="18,80")
        check = DataQualityCheck(
            name="age_range_check",
            description="Validate age is between 18 and 80",
            check_type=CheckType.VALIDATION,
            rule_id=rule.id,
            dataset_name="test_dataset"
        )
        
        mock_data_quality_check_repository.get_by_id.return_value = check
        mock_data_quality_rule_repository.get_by_id.return_value = rule
        
        # Act
        result = service.run_check(check.id, {"file_path": temp_csv_file})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result is not None
        assert result.result.total_records > 0

    def test_run_check_with_multiple_rules_and_conditions(self, mock_data_quality_check_repository, 
                                                        mock_data_quality_rule_repository, temp_csv_file):
        """Test check execution with complex rules having multiple conditions."""
        from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter
        from src.data_quality.application.services.rule_evaluator import RuleEvaluator
        from src.data_quality.domain.entities.data_quality_rule import RuleCondition, RuleOperator
        
        # Arrange
        adapter = PandasCSVAdapter()
        evaluator = RuleEvaluator()
        service = DataQualityCheckService(
            mock_data_quality_check_repository,
            mock_data_quality_rule_repository,
            adapter,
            evaluator
        )
        
        # Create complex rule with multiple conditions
        rule = DataQualityRule(
            name="complex_validation",
            description="Complex validation with multiple conditions",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset",
            logical_operator="AND"
        )
        
        # Add multiple conditions
        condition1 = RuleCondition(column_name="age", operator=RuleOperator.GREATER_THAN, value="18")
        condition2 = RuleCondition(column_name="age", operator=RuleOperator.LESS_THAN, value="80")
        condition3 = RuleCondition(column_name="name", operator=RuleOperator.NOT_EQUALS, value="")
        
        rule.add_condition(condition1)
        rule.add_condition(condition2)
        rule.add_condition(condition3)
        
        check = DataQualityCheck(
            name="complex_check",
            description="Complex validation check",
            check_type=CheckType.VALIDATION,
            rule_id=rule.id,
            dataset_name="test_dataset"
        )
        
        mock_data_quality_check_repository.get_by_id.return_value = check
        mock_data_quality_rule_repository.get_by_id.return_value = rule
        
        # Act
        result = service.run_check(check.id, {"file_path": temp_csv_file})
        
        # Assert
        assert result.status == CheckStatus.COMPLETED
        assert result.result is not None