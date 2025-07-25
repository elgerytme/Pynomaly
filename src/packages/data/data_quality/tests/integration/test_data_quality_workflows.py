"""Integration tests for data_quality package workflows."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.data_quality.domain.entities.data_profile import DataProfile, ProfileStatus
from src.data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckType, CheckStatus
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator
from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


@pytest.mark.integration
class TestDataQualityWorkflows:
    """Integration tests for complete data quality workflows."""

    @pytest.fixture
    def workflow_services(self, mock_data_profile_repository, mock_data_quality_check_repository, 
                         mock_data_quality_rule_repository):
        """Setup all services for workflow testing."""
        profiling_service = DataProfilingService(mock_data_profile_repository)
        rule_service = DataQualityRuleService(mock_data_quality_rule_repository)
        rule_evaluator = RuleEvaluator()
        adapter = PandasCSVAdapter()
        check_service = DataQualityCheckService(
            mock_data_quality_check_repository,
            mock_data_quality_rule_repository,
            adapter,
            rule_evaluator
        )
        
        return {
            'profiling': profiling_service,
            'rule': rule_service,
            'check': check_service,
            'evaluator': rule_evaluator,
            'adapter': adapter
        }

    @pytest.fixture
    def sample_dataset_file(self, tmp_path):
        """Create a sample dataset file for testing."""
        data = pd.DataFrame({
            'customer_id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'email': [f'customer{i}@example.com' if i % 10 != 0 else f'invalid_email_{i}' for i in range(1, 101)],
            'salary': np.random.uniform(30000, 120000, 100),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Unknown'], 100),
            'join_date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'is_active': np.random.choice([True, False], 100, p=[0.9, 0.1]),
            'performance_score': np.random.uniform(0.0, 5.0, 100)
        })
        
        # Introduce some data quality issues
        data.loc[5, 'name'] = None  # Missing name
        data.loc[10, 'age'] = -5   # Invalid age
        data.loc[15, 'salary'] = -1000  # Negative salary
        data.loc[20, 'department'] = ''  # Empty department
        
        file_path = tmp_path / "sample_data.csv"
        data.to_csv(file_path, index=False)
        return str(file_path)

    def test_complete_data_profiling_workflow(self, workflow_services, sample_dataset_file):
        """Test complete data profiling workflow from file to profile."""
        # Arrange
        profiling_service = workflow_services['profiling']
        adapter = workflow_services['adapter']
        dataset_name = "integration_test_dataset"
        source_config = {"file_path": sample_dataset_file}
        
        # Act
        profile = profiling_service.create_profile(dataset_name, adapter, source_config)
        
        # Assert
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == dataset_name
        assert profile.status == ProfileStatus.COMPLETED
        assert profile.total_rows == 100
        assert profile.total_columns == 9
        assert len(profile.column_profiles) == 9
        
        # Verify specific column profiles
        column_names = [cp.column_name for cp in profile.column_profiles]
        expected_columns = ['customer_id', 'name', 'age', 'email', 'salary', 'department', 'join_date', 'is_active', 'performance_score']
        assert all(col in column_names for col in expected_columns)
        
        # Verify statistics are calculated
        for col_profile in profile.column_profiles:
            assert col_profile.statistics is not None
            assert col_profile.statistics.total_count == 100

    def test_complete_rule_management_workflow(self, workflow_services):
        """Test complete rule management workflow."""
        # Arrange
        rule_service = workflow_services['rule']
        
        # Act & Assert - Create multiple rules
        rules_data = [
            {
                "name": "age_validation",
                "description": "Age should be between 18 and 80",
                "rule_type": RuleType.RANGE,
                "severity": RuleSeverity.ERROR,
                "dataset_name": "integration_test_dataset",
                "conditions": [{
                    "column_name": "age",
                    "operator": RuleOperator.BETWEEN,
                    "value": "18,80"
                }]
            },
            {
                "name": "email_format",
                "description": "Email should be valid format",
                "rule_type": RuleType.PATTERN,
                "severity": RuleSeverity.WARNING,
                "dataset_name": "integration_test_dataset",
                "conditions": [{
                    "column_name": "email",
                    "operator": RuleOperator.PATTERN,
                    "value": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }]
            },
            {
                "name": "salary_positive",
                "description": "Salary should be positive",
                "rule_type": RuleType.CUSTOM,
                "severity": RuleSeverity.ERROR,
                "dataset_name": "integration_test_dataset",
                "conditions": [{
                    "column_name": "salary",
                    "operator": RuleOperator.GREATER_THAN,
                    "value": "0"
                }]
            }
        ]
        
        created_rules = []
        for rule_data in rules_data:
            rule = rule_service.create_rule(rule_data)
            created_rules.append(rule)
        
        # Verify rules were created
        assert len(created_rules) == 3
        assert all(isinstance(rule, DataQualityRule) for rule in created_rules)
        
        # Test rule retrieval
        dataset_rules = rule_service.get_rules_by_dataset("integration_test_dataset")
        assert len(dataset_rules) >= 3  # At least our created rules
        
        # Test rule updates
        first_rule = created_rules[0]
        updated_rule = rule_service.update_rule(first_rule.id, {"description": "Updated age validation"})
        assert updated_rule.description == "Updated age validation"
        
        return created_rules

    def test_complete_data_quality_check_workflow(self, workflow_services, sample_dataset_file):
        """Test complete data quality check workflow."""
        # Arrange
        rule_service = workflow_services['rule']
        check_service = workflow_services['check']
        
        # First create rules
        rule_data = {
            "name": "integration_age_check",
            "description": "Age validation for integration test",
            "rule_type": RuleType.RANGE,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "integration_test_dataset"
        }
        
        created_rule = rule_service.create_rule(rule_data)
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.BETWEEN,
            value="18,80"
        )
        created_rule.add_condition(condition)
        
        # Create check
        check = DataQualityCheck(
            name="integration_age_validation_check",
            description="Integration test for age validation",
            check_type=CheckType.VALIDATION,
            rule_id=created_rule.id,
            dataset_name="integration_test_dataset"
        )
        
        # Mock repository responses
        check_service.data_quality_check_repository.get_by_id.return_value = check
        check_service.data_quality_rule_repository.get_by_id.return_value = created_rule
        
        # Act
        source_config = {"file_path": sample_dataset_file}
        result = check_service.run_check(check.id, source_config)
        
        # Assert
        assert isinstance(result, DataQualityCheck)
        assert result.status == CheckStatus.COMPLETED
        assert result.result is not None
        assert result.result.total_records == 100
        assert result.result.error_count >= 1  # Should catch the invalid age (-5)

    def test_end_to_end_data_quality_pipeline(self, workflow_services, sample_dataset_file):
        """Test complete end-to-end data quality pipeline."""
        # Arrange
        profiling_service = workflow_services['profiling']
        rule_service = workflow_services['rule']
        check_service = workflow_services['check']
        adapter = workflow_services['adapter']
        
        dataset_name = "e2e_pipeline_test"
        source_config = {"file_path": sample_dataset_file}
        
        # Step 1: Profile the data
        profile = profiling_service.create_profile(dataset_name, adapter, source_config)
        assert profile.status == ProfileStatus.COMPLETED
        
        # Step 2: Create rules based on profiling insights
        rules_data = [
            {
                "name": "name_not_null",
                "description": "Name should not be null",
                "rule_type": RuleType.NOT_NULL,
                "severity": RuleSeverity.ERROR,
                "dataset_name": dataset_name,
                "conditions": [{
                    "column_name": "name",
                    "operator": RuleOperator.NOT_EQUALS,
                    "value": "null"
                }]
            },
            {
                "name": "age_range_validation",
                "description": "Age should be reasonable",
                "rule_type": RuleType.RANGE,
                "severity": RuleSeverity.ERROR,
                "dataset_name": dataset_name,
                "conditions": [{
                    "column_name": "age",
                    "operator": RuleOperator.BETWEEN,
                    "value": "0,120"
                }]
            },
            {
                "name": "department_allowlist",
                "description": "Department should be valid",
                "rule_type": RuleType.CUSTOM,
                "severity": RuleSeverity.WARNING,
                "dataset_name": dataset_name,
                "conditions": [{
                    "column_name": "department",
                    "operator": RuleOperator.IN,
                    "value": "Engineering,Sales,Marketing,HR"
                }]
            }
        ]
        
        created_rules = []
        for rule_data in rules_data:
            rule = rule_service.create_rule(rule_data)
            created_rules.append(rule)
        
        # Step 3: Create and run checks for each rule
        check_results = []
        for rule in created_rules:
            # Create check
            check = DataQualityCheck(
                name=f"check_for_{rule.name}",
                description=f"Check for rule {rule.name}",
                check_type=CheckType.VALIDATION,
                rule_id=rule.id,
                dataset_name=dataset_name
            )
            
            # Mock repository responses
            check_service.data_quality_check_repository.get_by_id.return_value = check
            check_service.data_quality_rule_repository.get_by_id.return_value = rule
            
            # Run check
            result = check_service.run_check(check.id, source_config)
            check_results.append(result)
        
        # Step 4: Verify pipeline results
        assert len(check_results) == 3
        assert all(result.status == CheckStatus.COMPLETED for result in check_results)
        
        # Analyze results
        total_errors = sum(result.result.error_count for result in check_results)
        assert total_errors > 0  # Should find some data quality issues
        
        # Generate summary report
        summary = {
            "dataset_name": dataset_name,
            "total_records": profile.total_rows,
            "total_columns": profile.total_columns,
            "rules_executed": len(created_rules),
            "checks_completed": len(check_results),
            "total_errors_found": total_errors,
            "overall_quality_score": 1.0 - (total_errors / (profile.total_rows * len(created_rules)))
        }
        
        assert summary["overall_quality_score"] > 0.8  # Should have good overall quality
        
        return summary

    def test_batch_processing_workflow(self, workflow_services, tmp_path):
        """Test batch processing of multiple datasets."""
        # Arrange
        profiling_service = workflow_services['profiling']
        adapter = workflow_services['adapter']
        
        # Create multiple test datasets
        datasets = []
        for i in range(3):
            data = pd.DataFrame({
                'id': range(1, 51),
                'value': np.random.uniform(0, 100, 50),
                'category': np.random.choice(['A', 'B', 'C'], 50)
            })
            
            file_path = tmp_path / f"batch_dataset_{i}.csv"
            data.to_csv(file_path, index=False)
            datasets.append({
                'name': f"batch_dataset_{i}",
                'file_path': str(file_path)
            })
        
        # Act
        batch_results = []
        for dataset in datasets:
            source_config = {"file_path": dataset['file_path']}
            profile = profiling_service.create_profile(dataset['name'], adapter, source_config)
            batch_results.append(profile)
        
        # Assert
        assert len(batch_results) == 3
        assert all(profile.status == ProfileStatus.COMPLETED for profile in batch_results)
        assert all(profile.total_rows == 50 for profile in batch_results)
        assert all(profile.total_columns == 3 for profile in batch_results)

    def test_error_handling_in_workflow(self, workflow_services, tmp_path):
        """Test error handling throughout the workflow."""
        # Arrange
        profiling_service = workflow_services['profiling']
        rule_service = workflow_services['rule']
        adapter = workflow_services['adapter']
        
        # Test with non-existent file
        invalid_source_config = {"file_path": "/nonexistent/file.csv"}
        
        # Act & Assert - Should handle file not found gracefully
        with pytest.raises(FileNotFoundError):
            profiling_service.create_profile("invalid_dataset", adapter, invalid_source_config)
        
        # Test with invalid rule data
        invalid_rule_data = {
            "name": "",  # Invalid: empty name
            "description": "Invalid rule",
            "rule_type": "INVALID_TYPE",  # Invalid type
            "severity": RuleSeverity.ERROR,
            "dataset_name": "test_dataset"
        }
        
        with pytest.raises(ValueError):
            rule_service.create_rule(invalid_rule_data)

    def test_concurrent_workflow_execution(self, workflow_services, sample_dataset_file):
        """Test concurrent execution of workflows."""
        import threading
        import time
        
        # Arrange
        profiling_service = workflow_services['profiling']
        adapter = workflow_services['adapter']
        
        results = []
        errors = []
        
        def profile_thread(thread_id):
            try:
                dataset_name = f"concurrent_dataset_{thread_id}"
                source_config = {"file_path": sample_dataset_file}
                profile = profiling_service.create_profile(dataset_name, adapter, source_config)
                results.append(profile)
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for i in range(3):
            thread = threading.Thread(target=profile_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 3
        assert len(errors) == 0
        assert all(profile.status == ProfileStatus.COMPLETED for profile in results)

    def test_data_quality_metrics_aggregation(self, workflow_services, sample_dataset_file):
        """Test aggregation of data quality metrics across multiple checks."""
        # Arrange
        rule_service = workflow_services['rule']
        check_service = workflow_services['check']
        
        # Create multiple rules
        rules_data = [
            {
                "name": "metrics_rule_1",
                "description": "First metrics rule",
                "rule_type": RuleType.NOT_NULL,
                "severity": RuleSeverity.ERROR,
                "dataset_name": "metrics_test",
                "conditions": [{"column_name": "name", "operator": RuleOperator.NOT_EQUALS, "value": "null"}]
            },
            {
                "name": "metrics_rule_2", 
                "description": "Second metrics rule",
                "rule_type": RuleType.RANGE,
                "severity": RuleSeverity.WARNING,
                "dataset_name": "metrics_test",
                "conditions": [{"column_name": "age", "operator": RuleOperator.BETWEEN, "value": "0,120"}]
            }
        ]
        
        created_rules = []
        for rule_data in rules_data:
            rule = rule_service.create_rule(rule_data)
            # Add conditions
            for cond_data in rule_data["conditions"]:
                condition = RuleCondition(
                    column_name=cond_data["column_name"],
                    operator=cond_data["operator"],
                    value=cond_data["value"]
                )
                rule.add_condition(condition)
            created_rules.append(rule)
        
        # Run checks and collect metrics
        source_config = {"file_path": sample_dataset_file}
        check_results = []
        
        for rule in created_rules:
            check = DataQualityCheck(
                name=f"metrics_check_{rule.name}",
                description=f"Metrics check for {rule.name}",
                check_type=CheckType.VALIDATION,
                rule_id=rule.id,
                dataset_name="metrics_test"
            )
            
            # Mock repository responses
            check_service.data_quality_check_repository.get_by_id.return_value = check
            check_service.data_quality_rule_repository.get_by_id.return_value = rule
            
            result = check_service.run_check(check.id, source_config)
            check_results.append(result)
        
        # Aggregate metrics
        total_records_checked = sum(result.result.total_records for result in check_results)
        total_errors = sum(result.result.error_count for result in check_results)
        average_score = sum(result.result.score for result in check_results) / len(check_results)
        
        # Assert
        assert total_records_checked == 200  # 100 records * 2 rules
        assert total_errors >= 0
        assert 0.0 <= average_score <= 1.0

    def test_workflow_with_custom_adapters(self, workflow_services, tmp_path):
        """Test workflow with custom data source adapters."""
        # Create JSON test data
        import json
        
        json_data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.2},
            {"id": 3, "name": "Charlie", "score": 92.1}
        ]
        
        json_file = tmp_path / "test_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
        
        # Create custom JSON adapter (mock)
        class MockJSONAdapter:
            def read_data(self, source_config):
                with open(source_config['file_path'], 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        
        # Test profiling with custom adapter
        profiling_service = workflow_services['profiling']
        json_adapter = MockJSONAdapter()
        source_config = {"file_path": str(json_file)}
        
        profile = profiling_service.create_profile("json_dataset", json_adapter, source_config)
        
        # Assert
        assert profile.status == ProfileStatus.COMPLETED
        assert profile.total_rows == 3
        assert profile.total_columns == 3


@pytest.mark.integration
class TestDataQualityWorkflowsWithRealDatabase:
    """Integration tests with database persistence (if available)."""

    @pytest.fixture
    def database_services(self, in_memory_database):
        """Setup services with real database."""
        # This would use real repository implementations
        # For now, we'll skip if database models aren't available
        try:
            from src.data_quality.infrastructure.repositories.sqlalchemy_data_profile_repository import SQLAlchemyDataProfileRepository
            from src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_check_repository import SQLAlchemyDataQualityCheckRepository
            from src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_rule_repository import SQLAlchemyDataQualityRuleRepository
            
            profile_repo = SQLAlchemyDataProfileRepository(in_memory_database)
            check_repo = SQLAlchemyDataQualityCheckRepository(in_memory_database)
            rule_repo = SQLAlchemyDataQualityRuleRepository(in_memory_database)
            
            profiling_service = DataProfilingService(profile_repo)
            rule_service = DataQualityRuleService(rule_repo)
            rule_evaluator = RuleEvaluator()
            adapter = PandasCSVAdapter()
            check_service = DataQualityCheckService(check_repo, rule_repo, adapter, rule_evaluator)
            
            return {
                'profiling': profiling_service,
                'rule': rule_service,
                'check': check_service,
                'evaluator': rule_evaluator,
                'adapter': adapter
            }
        except ImportError:
            pytest.skip("Database repositories not available")

    @pytest.mark.database
    def test_persistent_workflow(self, database_services, sample_dataset_file):
        """Test workflow with database persistence."""
        # This test would verify that data is actually persisted
        # and can be retrieved across service calls
        pytest.skip("Database integration not fully implemented")

    @pytest.mark.database
    def test_workflow_with_transactions(self, database_services):
        """Test workflow with database transactions."""
        # This test would verify transaction handling
        pytest.skip("Database integration not fully implemented")