"""Comprehensive unit tests for DataQualityRuleService."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, call
from uuid import uuid4
from datetime import datetime

from src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator


@pytest.mark.unit
class TestDataQualityRuleService:
    """Test suite for DataQualityRuleService."""

    def test_create_rule_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule creation."""
        # Arrange
        rule_data = {
            "name": "test_rule",
            "description": "Test rule description",
            "rule_type": RuleType.NOT_NULL,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        data_quality_rule_service.data_quality_rule_repository.save.return_value = sample_data_quality_rule
        
        # Act
        result = data_quality_rule_service.create_rule(rule_data)
        
        # Assert
        assert isinstance(result, DataQualityRule)
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once()
        saved_rule = data_quality_rule_service.data_quality_rule_repository.save.call_args[0][0]
        assert saved_rule.name == rule_data["name"]
        assert saved_rule.description == rule_data["description"]
        assert saved_rule.rule_type == rule_data["rule_type"]
        assert saved_rule.severity == rule_data["severity"]
        assert saved_rule.dataset_name == rule_data["dataset_name"]

    def test_create_rule_with_conditions(self, data_quality_rule_service):
        """Test rule creation with conditions."""
        # Arrange
        rule_data = {
            "name": "range_rule",
            "description": "Range validation rule",
            "rule_type": RuleType.RANGE,
            "severity": RuleSeverity.WARNING,
            "dataset_name": "test_dataset",
            "conditions": [
                {
                    "column_name": "age",
                    "operator": RuleOperator.BETWEEN,
                    "value": "18,80"
                },
                {
                    "column_name": "salary",
                    "operator": RuleOperator.GREATER_THAN,
                    "value": "0"
                }
            ]
        }
        
        # Act
        result = data_quality_rule_service.create_rule(rule_data)
        
        # Assert
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once()
        saved_rule = data_quality_rule_service.data_quality_rule_repository.save.call_args[0][0]
        assert len(saved_rule.conditions) == 2
        assert saved_rule.conditions[0].column_name == "age"
        assert saved_rule.conditions[0].operator == RuleOperator.BETWEEN
        assert saved_rule.conditions[1].column_name == "salary"
        assert saved_rule.conditions[1].operator == RuleOperator.GREATER_THAN

    def test_create_rule_missing_required_fields(self, data_quality_rule_service):
        """Test rule creation with missing required fields."""
        # Arrange
        incomplete_rule_data = {
            "name": "incomplete_rule",
            # Missing required fields
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Missing required field"):
            data_quality_rule_service.create_rule(incomplete_rule_data)

    def test_create_rule_invalid_rule_type(self, data_quality_rule_service):
        """Test rule creation with invalid rule type."""
        # Arrange
        rule_data = {
            "name": "invalid_rule",
            "description": "Invalid rule",
            "rule_type": "INVALID_TYPE",
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid rule type"):
            data_quality_rule_service.create_rule(rule_data)

    def test_create_rule_repository_error(self, data_quality_rule_service):
        """Test rule creation when repository save fails."""
        # Arrange
        rule_data = {
            "name": "test_rule",
            "description": "Test rule",
            "rule_type": RuleType.NOT_NULL,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        data_quality_rule_service.data_quality_rule_repository.save.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            data_quality_rule_service.create_rule(rule_data)

    def test_get_rule_by_id_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule retrieval by ID."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        
        # Act
        result = data_quality_rule_service.get_rule_by_id(rule_id)
        
        # Assert
        assert result == sample_data_quality_rule
        data_quality_rule_service.data_quality_rule_repository.get_by_id.assert_called_once_with(rule_id)

    def test_get_rule_by_id_not_found(self, data_quality_rule_service):
        """Test rule retrieval when rule is not found."""
        # Arrange
        rule_id = uuid4()
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="DataQualityRule with id .* not found"):
            data_quality_rule_service.get_rule_by_id(rule_id)

    def test_get_rules_by_dataset_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rules retrieval by dataset name."""
        # Arrange
        dataset_name = "test_dataset"
        rules = [sample_data_quality_rule]
        data_quality_rule_service.data_quality_rule_repository.get_by_dataset_name.return_value = rules
        
        # Act
        result = data_quality_rule_service.get_rules_by_dataset(dataset_name)
        
        # Assert
        assert result == rules
        data_quality_rule_service.data_quality_rule_repository.get_by_dataset_name.assert_called_once_with(dataset_name)

    def test_get_rules_by_dataset_empty_result(self, data_quality_rule_service):
        """Test rules retrieval when no rules exist for dataset."""
        # Arrange
        dataset_name = "nonexistent_dataset"
        data_quality_rule_service.data_quality_rule_repository.get_by_dataset_name.return_value = []
        
        # Act
        result = data_quality_rule_service.get_rules_by_dataset(dataset_name)
        
        # Assert
        assert result == []

    def test_update_rule_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule update."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        update_data = {
            "description": "Updated description",
            "severity": RuleSeverity.CRITICAL
        }
        
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        data_quality_rule_service.data_quality_rule_repository.save.return_value = sample_data_quality_rule
        
        # Act
        result = data_quality_rule_service.update_rule(rule_id, update_data)
        
        # Assert
        assert result.description == update_data["description"]
        assert result.severity == update_data["severity"]
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once_with(result)

    def test_update_rule_not_found(self, data_quality_rule_service):
        """Test rule update when rule is not found."""
        # Arrange
        rule_id = uuid4()
        update_data = {"description": "Updated description"}
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="DataQualityRule with id .* not found"):
            data_quality_rule_service.update_rule(rule_id, update_data)

    def test_update_rule_invalid_field(self, data_quality_rule_service, sample_data_quality_rule):
        """Test rule update with invalid field."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        update_data = {"invalid_field": "value"}
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid field"):
            data_quality_rule_service.update_rule(rule_id, update_data)

    def test_delete_rule_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule deletion."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        
        # Act
        data_quality_rule_service.delete_rule(rule_id)
        
        # Assert
        data_quality_rule_service.data_quality_rule_repository.delete.assert_called_once_with(rule_id)

    def test_delete_rule_not_found(self, data_quality_rule_service):
        """Test rule deletion when rule is not found."""
        # Arrange
        rule_id = uuid4()
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="DataQualityRule with id .* not found"):
            data_quality_rule_service.delete_rule(rule_id)

    def test_validate_rule_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule validation."""
        # Arrange
        rule_data = {
            "name": "valid_rule",
            "description": "Valid rule description",
            "rule_type": RuleType.NOT_NULL,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        # Act
        is_valid, errors = data_quality_rule_service.validate_rule(rule_data)
        
        # Assert
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_rule_missing_name(self, data_quality_rule_service):
        """Test rule validation with missing name."""
        # Arrange
        rule_data = {
            "description": "Rule without name",
            "rule_type": RuleType.NOT_NULL,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        # Act
        is_valid, errors = data_quality_rule_service.validate_rule(rule_data)
        
        # Assert
        assert is_valid is False
        assert "name" in str(errors)

    def test_validate_rule_invalid_conditions(self, data_quality_rule_service):
        """Test rule validation with invalid conditions."""
        # Arrange
        rule_data = {
            "name": "rule_with_invalid_conditions",
            "description": "Rule with invalid conditions",
            "rule_type": RuleType.RANGE,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset",
            "conditions": [
                {
                    "column_name": "",  # Invalid: empty column name
                    "operator": RuleOperator.GREATER_THAN,
                    "value": "10"
                }
            ]
        }
        
        # Act
        is_valid, errors = data_quality_rule_service.validate_rule(rule_data)
        
        # Assert
        assert is_valid is False
        assert "column_name" in str(errors)

    def test_get_rules_by_type_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rules retrieval by type."""
        # Arrange
        rule_type = RuleType.NOT_NULL
        rules = [sample_data_quality_rule]
        data_quality_rule_service.data_quality_rule_repository.get_by_type.return_value = rules
        
        # Act
        result = data_quality_rule_service.get_rules_by_type(rule_type)
        
        # Assert
        assert result == rules
        data_quality_rule_service.data_quality_rule_repository.get_by_type.assert_called_once_with(rule_type)

    def test_get_rules_by_severity_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rules retrieval by severity."""
        # Arrange
        severity = RuleSeverity.ERROR
        rules = [sample_data_quality_rule]
        data_quality_rule_service.data_quality_rule_repository.get_by_severity.return_value = rules
        
        # Act
        result = data_quality_rule_service.get_rules_by_severity(severity)
        
        # Assert
        assert result == rules
        data_quality_rule_service.data_quality_rule_repository.get_by_severity.assert_called_once_with(severity)

    def test_duplicate_rule_detection(self, data_quality_rule_service, sample_data_quality_rule):
        """Test duplicate rule detection."""
        # Arrange
        rule_data = {
            "name": sample_data_quality_rule.name,
            "description": "Duplicate rule",
            "rule_type": sample_data_quality_rule.rule_type,
            "severity": RuleSeverity.ERROR,
            "dataset_name": sample_data_quality_rule.dataset_name
        }
        
        data_quality_rule_service.data_quality_rule_repository.get_by_name_and_dataset.return_value = sample_data_quality_rule
        
        # Act & Assert
        with pytest.raises(ValueError, match="Rule with name .* already exists"):
            data_quality_rule_service.create_rule(rule_data)

    def test_export_rules_to_json(self, data_quality_rule_service, sample_data_quality_rule):
        """Test exporting rules to JSON format."""
        # Arrange
        rules = [sample_data_quality_rule]
        data_quality_rule_service.data_quality_rule_repository.get_all.return_value = rules
        
        # Act
        json_export = data_quality_rule_service.export_rules_to_json()
        
        # Assert
        assert isinstance(json_export, str)
        assert sample_data_quality_rule.name in json_export
        assert sample_data_quality_rule.rule_type.value in json_export

    def test_import_rules_from_json_success(self, data_quality_rule_service):
        """Test importing rules from JSON format."""
        # Arrange
        json_data = '''[
            {
                "name": "imported_rule",
                "description": "Imported rule description",
                "rule_type": "NOT_NULL",
                "severity": "ERROR",
                "dataset_name": "test_dataset",
                "conditions": [
                    {
                        "column_name": "id",
                        "operator": "NOT_EQUALS",
                        "value": "null"
                    }
                ]
            }
        ]'''
        
        # Act
        imported_rules = data_quality_rule_service.import_rules_from_json(json_data)
        
        # Assert
        assert len(imported_rules) == 1
        assert imported_rules[0].name == "imported_rule"
        assert imported_rules[0].rule_type == RuleType.NOT_NULL
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once()

    def test_import_rules_from_json_invalid_format(self, data_quality_rule_service):
        """Test importing rules from invalid JSON format."""
        # Arrange
        invalid_json = "invalid json format"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid JSON format"):
            data_quality_rule_service.import_rules_from_json(invalid_json)

    def test_clone_rule_success(self, data_quality_rule_service, sample_data_quality_rule):
        """Test successful rule cloning."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        new_name = "cloned_rule"
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        
        # Act
        cloned_rule = data_quality_rule_service.clone_rule(rule_id, new_name)
        
        # Assert
        assert cloned_rule.name == new_name
        assert cloned_rule.description == sample_data_quality_rule.description
        assert cloned_rule.rule_type == sample_data_quality_rule.rule_type
        assert cloned_rule.id != sample_data_quality_rule.id
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once_with(cloned_rule)

    @pytest.mark.parametrize("rule_type,expected_validation", [
        (RuleType.NOT_NULL, True),
        (RuleType.UNIQUE, True),
        (RuleType.RANGE, True),
        (RuleType.PATTERN, True),
        (RuleType.CUSTOM, True),
    ])
    def test_create_rule_different_types(self, data_quality_rule_service, rule_type, expected_validation):
        """Test rule creation with different rule types."""
        # Arrange
        rule_data = {
            "name": f"{rule_type.value}_rule",
            "description": f"Test rule for {rule_type.value}",
            "rule_type": rule_type,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        # Act
        result = data_quality_rule_service.create_rule(rule_data)
        
        # Assert
        data_quality_rule_service.data_quality_rule_repository.save.assert_called_once()
        saved_rule = data_quality_rule_service.data_quality_rule_repository.save.call_args[0][0]
        assert saved_rule.rule_type == rule_type

    def test_get_rule_statistics(self, data_quality_rule_service):
        """Test getting rule statistics."""
        # Arrange
        rules = []
        for rule_type in RuleType:
            for severity in RuleSeverity:
                rule = DataQualityRule(
                    name=f"{rule_type.value}_{severity.value}_rule",
                    description="Test rule",
                    rule_type=rule_type,
                    severity=severity,
                    dataset_name="test_dataset"
                )
                rules.append(rule)
        
        data_quality_rule_service.data_quality_rule_repository.get_all.return_value = rules
        
        # Act
        stats = data_quality_rule_service.get_rule_statistics()
        
        # Assert
        assert "total_rules" in stats
        assert "by_type" in stats
        assert "by_severity" in stats
        assert "by_dataset" in stats
        assert stats["total_rules"] == len(rules)

    def test_bulk_create_rules_success(self, data_quality_rule_service):
        """Test bulk rule creation."""
        # Arrange
        rules_data = [
            {
                "name": "bulk_rule_1",
                "description": "First bulk rule",
                "rule_type": RuleType.NOT_NULL,
                "severity": RuleSeverity.ERROR,
                "dataset_name": "test_dataset"
            },
            {
                "name": "bulk_rule_2",
                "description": "Second bulk rule",
                "rule_type": RuleType.UNIQUE,
                "severity": RuleSeverity.WARNING,
                "dataset_name": "test_dataset"
            }
        ]
        
        # Act
        created_rules = data_quality_rule_service.bulk_create_rules(rules_data)
        
        # Assert
        assert len(created_rules) == 2
        assert data_quality_rule_service.data_quality_rule_repository.save.call_count == 2

    def test_bulk_create_rules_partial_failure(self, data_quality_rule_service):
        """Test bulk rule creation with partial failures."""
        # Arrange
        rules_data = [
            {
                "name": "valid_rule",
                "description": "Valid rule",
                "rule_type": RuleType.NOT_NULL,
                "severity": RuleSeverity.ERROR,
                "dataset_name": "test_dataset"
            },
            {
                "name": "",  # Invalid: empty name
                "description": "Invalid rule",
                "rule_type": RuleType.UNIQUE,
                "severity": RuleSeverity.WARNING,
                "dataset_name": "test_dataset"
            }
        ]
        
        # Act
        created_rules, errors = data_quality_rule_service.bulk_create_rules(rules_data, fail_on_error=False)
        
        # Assert
        assert len(created_rules) == 1
        assert len(errors) == 1
        assert "name" in str(errors[0])

    def test_concurrent_rule_operations(self, mock_data_quality_rule_repository):
        """Test concurrent rule operations."""
        import threading
        import time
        
        # Arrange
        service = DataQualityRuleService(mock_data_quality_rule_repository)
        results = []
        errors = []
        
        def create_rule_thread(thread_id):
            try:
                rule_data = {
                    "name": f"concurrent_rule_{thread_id}",
                    "description": f"Concurrent rule {thread_id}",
                    "rule_type": RuleType.NOT_NULL,
                    "severity": RuleSeverity.ERROR,
                    "dataset_name": "test_dataset"
                }
                result = service.create_rule(rule_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_rule_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 5
        assert len(errors) == 0
        assert mock_data_quality_rule_repository.save.call_count == 5

    def test_rule_versioning(self, data_quality_rule_service, sample_data_quality_rule):
        """Test rule versioning functionality."""
        # Arrange
        rule_id = sample_data_quality_rule.id
        original_version = sample_data_quality_rule.version if hasattr(sample_data_quality_rule, 'version') else 1
        update_data = {"description": "Updated description"}
        
        data_quality_rule_service.data_quality_rule_repository.get_by_id.return_value = sample_data_quality_rule
        
        # Act
        updated_rule = data_quality_rule_service.update_rule(rule_id, update_data)
        
        # Assert
        # Verify that version was incremented (if versioning is implemented)
        if hasattr(updated_rule, 'version'):
            assert updated_rule.version > original_version


@pytest.mark.integration
class TestDataQualityRuleServiceIntegration:
    """Integration tests for DataQualityRuleService."""

    def test_rule_lifecycle_integration(self, mock_data_quality_rule_repository):
        """Test complete rule lifecycle integration."""
        # Arrange
        service = DataQualityRuleService(mock_data_quality_rule_repository)
        
        # Mock repository responses for lifecycle
        created_rule = DataQualityRule(
            name="lifecycle_rule",
            description="Rule for lifecycle test",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        
        mock_data_quality_rule_repository.save.return_value = created_rule
        mock_data_quality_rule_repository.get_by_id.return_value = created_rule
        
        # Act & Assert - Create
        rule_data = {
            "name": "lifecycle_rule",
            "description": "Rule for lifecycle test",
            "rule_type": RuleType.NOT_NULL,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        created = service.create_rule(rule_data)
        assert created.name == "lifecycle_rule"
        
        # Act & Assert - Read
        retrieved = service.get_rule_by_id(created.id)
        assert retrieved.name == created.name
        
        # Act & Assert - Update
        updated = service.update_rule(created.id, {"description": "Updated description"})
        assert updated.description == "Updated description"
        
        # Act & Assert - Delete
        service.delete_rule(created.id)
        mock_data_quality_rule_repository.delete.assert_called_once_with(created.id)

    def test_rule_with_complex_conditions_integration(self, mock_data_quality_rule_repository):
        """Test rule creation and validation with complex conditions."""
        # Arrange
        service = DataQualityRuleService(mock_data_quality_rule_repository)
        
        rule_data = {
            "name": "complex_validation_rule",
            "description": "Complex rule with multiple conditions",
            "rule_type": RuleType.CUSTOM,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset",
            "logical_operator": "AND",
            "conditions": [
                {
                    "column_name": "age",
                    "operator": RuleOperator.BETWEEN,
                    "value": "18,65"
                },
                {
                    "column_name": "email",
                    "operator": RuleOperator.PATTERN,
                    "value": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                {
                    "column_name": "salary",
                    "operator": RuleOperator.GREATER_THAN,
                    "value": "0"
                }
            ]
        }
        
        # Act
        created_rule = service.create_rule(rule_data)
        
        # Assert
        mock_data_quality_rule_repository.save.assert_called_once()
        saved_rule = mock_data_quality_rule_repository.save.call_args[0][0]
        assert len(saved_rule.conditions) == 3
        assert saved_rule.logical_operator == "AND"
        assert all(isinstance(condition, RuleCondition) for condition in saved_rule.conditions)