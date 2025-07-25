"""Performance tests for data_quality package with large datasets."""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckType
from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator
from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


@pytest.mark.performance
class TestDataQualityPerformance:
    """Performance tests for data quality operations."""

    @pytest.fixture
    def performance_monitor(self):
        """Performance monitoring fixture."""
        class PerformanceMonitor:
            def __init__(self):
                self.start_time = None
                self.start_memory = None
                self.process = psutil.Process(os.getpid())
            
            def start(self):
                self.start_time = time.perf_counter()
                self.start_memory = self.process.memory_info().rss
            
            def stop(self):
                end_time = time.perf_counter()
                end_memory = self.process.memory_info().rss
                
                return {
                    'duration_seconds': end_time - self.start_time,
                    'memory_delta_mb': (end_memory - self.start_memory) / 1024 / 1024,
                    'peak_memory_mb': end_memory / 1024 / 1024
                }
        
        return PerformanceMonitor()

    @pytest.fixture(params=[1000, 10000, 50000, 100000])
    def large_dataset(self, request, tmp_path):
        """Create large datasets of varying sizes."""
        size = request.param
        np.random.seed(42)  # For reproducible results
        
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'name': [f'Person_{i}' for i in range(1, size + 1)],
            'age': np.random.randint(18, 80, size),
            'email': [f'person{i}@example.com' for i in range(1, size + 1)],
            'salary': np.random.uniform(30000, 120000, size),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], size),
            'join_date': pd.date_range('2020-01-01', periods=size, freq='H'),
            'is_active': np.random.choice([True, False], size, p=[0.8, 0.2]),
            'rating': np.random.uniform(1.0, 5.0, size),
            'description': [f'Description for person {i} with some longer text content' for i in range(1, size + 1)]
        })
        
        # Add some data quality issues proportionally
        issue_count = max(1, size // 100)  # 1% of records have issues
        issue_indices = np.random.choice(size, issue_count, replace=False)
        
        for idx in issue_indices:
            if idx % 4 == 0:
                data.loc[idx, 'name'] = None  # Missing name
            elif idx % 4 == 1:
                data.loc[idx, 'age'] = -5     # Invalid age
            elif idx % 4 == 2:
                data.loc[idx, 'salary'] = -1000  # Negative salary
            else:
                data.loc[idx, 'email'] = 'invalid_email'  # Invalid email
        
        file_path = tmp_path / f"large_dataset_{size}.csv"
        data.to_csv(file_path, index=False)
        
        return {
            'size': size,
            'file_path': str(file_path),
            'data': data,
            'expected_issues': issue_count
        }

    @pytest.fixture
    def performance_services(self, mock_data_profile_repository, 
                           mock_data_quality_check_repository,
                           mock_data_quality_rule_repository):
        """Setup services for performance testing."""
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

    @pytest.mark.slow
    def test_data_profiling_performance(self, performance_services, large_dataset, performance_monitor):
        """Test data profiling performance with large datasets."""
        # Arrange
        profiling_service = performance_services['profiling']
        adapter = performance_services['adapter']
        dataset_name = f"perf_test_dataset_{large_dataset['size']}"
        source_config = {"file_path": large_dataset['file_path']}
        
        # Act
        performance_monitor.start()
        profile = profiling_service.create_profile(dataset_name, adapter, source_config)
        metrics = performance_monitor.stop()
        
        # Assert
        assert profile is not None
        assert profile.total_rows == large_dataset['size']
        assert profile.total_columns == 10
        
        # Performance assertions based on dataset size
        size = large_dataset['size']
        if size <= 1000:
            assert metrics['duration_seconds'] < 2.0
            assert metrics['memory_delta_mb'] < 20
        elif size <= 10000:
            assert metrics['duration_seconds'] < 10.0
            assert metrics['memory_delta_mb'] < 50
        elif size <= 50000:
            assert metrics['duration_seconds'] < 30.0
            assert metrics['memory_delta_mb'] < 150
        else:  # 100000+
            assert metrics['duration_seconds'] < 60.0
            assert metrics['memory_delta_mb'] < 300

    @pytest.mark.slow
    def test_rule_evaluation_performance(self, performance_services, large_dataset, performance_monitor):
        """Test rule evaluation performance with large datasets."""
        # Arrange
        rule_service = performance_services['rule']
        evaluator = performance_services['evaluator']
        
        # Create a complex rule
        rule_data = {
            "name": "performance_validation_rule",
            "description": "Complex validation rule for performance testing",
            "rule_type": RuleType.CUSTOM,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "performance_test",
            "logical_operator": "AND"
        }
        
        rule = rule_service.create_rule(rule_data)
        
        # Add multiple conditions
        conditions = [
            RuleCondition(column_name="age", operator=RuleOperator.BETWEEN, value="18,80"),
            RuleCondition(column_name="salary", operator=RuleOperator.GREATER_THAN, value="0"),
            RuleCondition(column_name="name", operator=RuleOperator.NOT_EQUALS, value="null"),
            RuleCondition(column_name="email", operator=RuleOperator.PATTERN, 
                         value=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        ]
        
        for condition in conditions:
            rule.add_condition(condition)
        
        # Act - Evaluate rule against all records
        performance_monitor.start()
        
        results = []
        for _, row in large_dataset['data'].iterrows():
            record = row.to_dict()
            result = evaluator.evaluate_record(record, rule)
            results.append(result)
        
        metrics = performance_monitor.stop()
        
        # Assert
        assert len(results) == large_dataset['size']
        failed_count = sum(1 for r in results if not r)
        
        # Should detect the data quality issues we introduced
        assert failed_count >= large_dataset['expected_issues'] * 0.5  # At least half should be caught
        
        # Performance assertions
        size = large_dataset['size']
        records_per_second = size / metrics['duration_seconds']
        
        if size <= 1000:
            assert records_per_second > 500
        elif size <= 10000:
            assert records_per_second > 200
        elif size <= 50000:
            assert records_per_second > 100
        else:  # 100000+
            assert records_per_second > 50

    @pytest.mark.slow
    def test_data_quality_check_performance(self, performance_services, large_dataset, performance_monitor):
        """Test complete data quality check performance."""
        # Arrange
        rule_service = performance_services['rule']
        check_service = performance_services['check']
        
        # Create rule
        rule_data = {
            "name": "perf_check_rule",
            "description": "Performance check rule",
            "rule_type": RuleType.RANGE,
            "severity": RuleSeverity.ERROR,
            "dataset_name": "perf_check_dataset"
        }
        
        rule = rule_service.create_rule(rule_data)
        condition = RuleCondition(
            column_name="age",
            operator=RuleOperator.BETWEEN,
            value="0,120"
        )
        rule.add_condition(condition)
        
        # Create check
        check = DataQualityCheck(
            name="performance_check",
            description="Performance test check",
            check_type=CheckType.VALIDATION,
            rule_id=rule.id,
            dataset_name="perf_check_dataset"
        )
        
        # Mock repository responses
        check_service.data_quality_check_repository.get_by_id.return_value = check
        check_service.data_quality_rule_repository.get_by_id.return_value = rule
        
        # Act
        performance_monitor.start()
        source_config = {"file_path": large_dataset['file_path']}
        result = check_service.run_check(check.id, source_config)
        metrics = performance_monitor.stop()
        
        # Assert
        assert result is not None
        assert result.result.total_records == large_dataset['size']
        
        # Performance assertions
        size = large_dataset['size']
        if size <= 1000:
            assert metrics['duration_seconds'] < 3.0
        elif size <= 10000:
            assert metrics['duration_seconds'] < 15.0
        elif size <= 50000:
            assert metrics['duration_seconds'] < 45.0
        else:  # 100000+
            assert metrics['duration_seconds'] < 90.0

    @pytest.mark.slow
    def test_memory_efficiency_with_large_datasets(self, performance_services, tmp_path):
        """Test memory efficiency when processing very large datasets."""
        # Create a very large dataset
        size = 200000
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'value': np.random.uniform(0, 1000, size),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
            'description': [f'Long description text for record {i} with additional content' * 5 for i in range(1, size + 1)]
        })
        
        file_path = tmp_path / "memory_test_dataset.csv"
        data.to_csv(file_path, index=False)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Profile the large dataset
        profiling_service = performance_services['profiling']
        adapter = performance_services['adapter']
        source_config = {"file_path": str(file_path)}
        
        profile = profiling_service.create_profile("memory_test", adapter, source_config)
        
        peak_memory = process.memory_info().rss
        memory_increase_mb = (peak_memory - initial_memory) / 1024 / 1024
        
        # Assert
        assert profile.total_rows == size
        assert memory_increase_mb < 1000  # Should not use more than 1GB additional memory

    @pytest.mark.slow
    def test_concurrent_processing_performance(self, performance_services, tmp_path, performance_monitor):
        """Test performance with concurrent processing."""
        import threading
        import concurrent.futures
        
        # Create multiple datasets
        datasets = []
        for i in range(5):
            size = 5000
            data = pd.DataFrame({
                'id': range(1, size + 1),
                'value': np.random.uniform(0, 100, size),
                'category': np.random.choice(['A', 'B', 'C'], size)
            })
            
            file_path = tmp_path / f"concurrent_dataset_{i}.csv"
            data.to_csv(file_path, index=False)
            datasets.append(str(file_path))
        
        profiling_service = performance_services['profiling']
        adapter = performance_services['adapter']
        
        def profile_dataset(file_path, dataset_id):
            source_config = {"file_path": file_path}
            return profiling_service.create_profile(f"concurrent_dataset_{dataset_id}", adapter, source_config)
        
        # Test sequential processing
        performance_monitor.start()
        sequential_results = []
        for i, dataset_path in enumerate(datasets):
            result = profile_dataset(dataset_path, i)
            sequential_results.append(result)
        sequential_metrics = performance_monitor.stop()
        
        # Test concurrent processing
        performance_monitor.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_dataset = {
                executor.submit(profile_dataset, path, i): i 
                for i, path in enumerate(datasets)
            }
            concurrent_results = []
            for future in concurrent.futures.as_completed(future_to_dataset):
                result = future.result()
                concurrent_results.append(result)
        concurrent_metrics = performance_monitor.stop()
        
        # Assert
        assert len(sequential_results) == 5
        assert len(concurrent_results) == 5
        assert all(result.total_rows == 5000 for result in sequential_results)
        assert all(result.total_rows == 5000 for result in concurrent_results)
        
        # Concurrent processing should be faster (with some tolerance for overhead)
        speedup_ratio = sequential_metrics['duration_seconds'] / concurrent_metrics['duration_seconds']
        assert speedup_ratio > 1.2  # At least 20% faster

    @pytest.mark.slow
    def test_pattern_matching_performance(self, performance_services, tmp_path, performance_monitor):
        """Test performance of complex pattern matching rules."""
        # Create dataset with various text patterns
        size = 20000
        emails = [f'user{i}@example.com' if i % 10 != 0 else f'invalid_email_{i}' for i in range(size)]
        phones = [f'+1-555-{i:03d}-{(i*7)%10000:04d}' if i % 15 != 0 else f'invalid_phone_{i}' for i in range(size)]
        
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'email': emails,
            'phone': phones,
            'ssn': [f'{i:03d}-{(i*2)%100:02d}-{(i*3)%10000:04d}' for i in range(size)]
        })
        
        file_path = tmp_path / "pattern_test_dataset.csv"
        data.to_csv(file_path, index=False)
        
        # Create complex pattern matching rules
        rule_service = performance_services['rule']
        check_service = performance_services['check']
        
        rules_data = [
            {
                "name": "email_pattern_check",
                "rule_type": RuleType.PATTERN,
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "column": "email"
            },
            {
                "name": "phone_pattern_check", 
                "rule_type": RuleType.PATTERN,
                "pattern": r"^\+1-\d{3}-\d{3}-\d{4}$",
                "column": "phone"
            },
            {
                "name": "ssn_pattern_check",
                "rule_type": RuleType.PATTERN,
                "pattern": r"^\d{3}-\d{2}-\d{4}$",
                "column": "ssn"
            }
        ]
        
        # Test each pattern rule
        performance_monitor.start()
        
        for rule_data in rules_data:
            rule = DataQualityRule(
                name=rule_data["name"],
                description=f"Pattern check for {rule_data['column']}",
                rule_type=rule_data["rule_type"],
                severity=RuleSeverity.WARNING,
                dataset_name="pattern_test"
            )
            
            condition = RuleCondition(
                column_name=rule_data["column"],
                operator=RuleOperator.PATTERN,
                value=rule_data["pattern"]
            )
            rule.add_condition(condition)
            
            # Create and run check
            check = DataQualityCheck(
                name=f"check_{rule_data['name']}",
                description=f"Check for {rule_data['name']}",
                check_type=CheckType.VALIDATION,
                rule_id=rule.id,
                dataset_name="pattern_test"
            )
            
            check_service.data_quality_check_repository.get_by_id.return_value = check
            check_service.data_quality_rule_repository.get_by_id.return_value = rule
            
            source_config = {"file_path": str(file_path)}
            result = check_service.run_check(check.id, source_config)
            
            assert result.result.total_records == size
        
        metrics = performance_monitor.stop()
        
        # Assert performance
        total_pattern_checks = size * len(rules_data)
        checks_per_second = total_pattern_checks / metrics['duration_seconds']
        assert checks_per_second > 1000  # Should process at least 1000 pattern checks per second

    @pytest.mark.slow
    def test_batch_processing_performance(self, performance_services, tmp_path, performance_monitor):
        """Test performance of batch processing multiple datasets."""
        # Create multiple datasets of different sizes
        dataset_configs = [
            {'size': 1000, 'name': 'small_batch'},
            {'size': 5000, 'name': 'medium_batch'},  
            {'size': 10000, 'name': 'large_batch'},
            {'size': 2500, 'name': 'mixed_batch_1'},
            {'size': 7500, 'name': 'mixed_batch_2'}
        ]
        
        datasets = []
        for config in dataset_configs:
            size = config['size']
            data = pd.DataFrame({
                'id': range(1, size + 1),
                'value': np.random.uniform(0, 1000, size),
                'category': np.random.choice(['A', 'B', 'C', 'D'], size),
                'status': np.random.choice(['active', 'inactive'], size, p=[0.8, 0.2])
            })
            
            file_path = tmp_path / f"{config['name']}.csv"
            data.to_csv(file_path, index=False)
            
            datasets.append({
                'name': config['name'],
                'file_path': str(file_path),
                'size': size
            })
        
        # Batch process all datasets
        profiling_service = performance_services['profiling']
        adapter = performance_services['adapter']
        
        performance_monitor.start()
        
        batch_results = []
        for dataset in datasets:
            source_config = {"file_path": dataset['file_path']}
            profile = profiling_service.create_profile(dataset['name'], adapter, source_config)
            batch_results.append(profile)
        
        metrics = performance_monitor.stop()
        
        # Assert
        assert len(batch_results) == len(datasets)
        total_records = sum(dataset['size'] for dataset in datasets)
        assert sum(profile.total_rows for profile in batch_results) == total_records
        
        # Performance assertion
        records_per_second = total_records / metrics['duration_seconds']
        assert records_per_second > 500  # Should process at least 500 records per second in batch

    @pytest.mark.slow
    def test_scalability_limits(self, performance_services, tmp_path):
        """Test system behavior at scalability limits."""
        # Test with progressively larger datasets to find limits
        sizes = [10000, 50000, 100000, 250000]
        profiling_service = performance_services['profiling']
        adapter = performance_services['adapter']
        
        results = []
        for size in sizes:
            try:
                # Create dataset
                data = pd.DataFrame({
                    'id': range(1, size + 1),
                    'value': np.random.uniform(0, 1000, size)
                })
                
                file_path = tmp_path / f"scalability_test_{size}.csv"
                data.to_csv(file_path, index=False)
                
                # Time the profiling
                start_time = time.perf_counter()
                source_config = {"file_path": str(file_path)}
                profile = profiling_service.create_profile(f"scalability_test_{size}", adapter, source_config)
                end_time = time.perf_counter()
                
                duration = end_time - start_time
                throughput = size / duration
                
                results.append({
                    'size': size,
                    'duration': duration,
                    'throughput': throughput,
                    'success': True
                })
                
                # Cleanup
                os.unlink(file_path)
                
            except Exception as e:
                results.append({
                    'size': size,
                    'error': str(e),
                    'success': False
                })
                break
        
        # Assert that we can handle reasonable dataset sizes
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) >= 2  # Should handle at least 2 size categories
        
        # Verify throughput doesn't degrade too much
        if len(successful_results) > 1:
            first_throughput = successful_results[0]['throughput']
            last_throughput = successful_results[-1]['throughput']
            degradation_ratio = first_throughput / last_throughput
            assert degradation_ratio < 10  # Throughput shouldn't degrade by more than 10x


@pytest.mark.performance
class TestDataQualityMemoryLeaks:
    """Test for memory leaks in data quality operations."""

    @pytest.mark.slow
    def test_memory_leak_in_repeated_profiling(self, mock_data_profile_repository, tmp_path):
        """Test for memory leaks in repeated profiling operations."""
        profiling_service = DataProfilingService(mock_data_profile_repository)
        adapter = PandasCSVAdapter()
        
        # Create test dataset
        size = 5000
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'value': np.random.uniform(0, 100, size)
        })
        
        file_path = tmp_path / "memory_leak_test.csv"
        data.to_csv(file_path, index=False)
        source_config = {"file_path": str(file_path)}
        
        # Monitor memory over multiple iterations
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        memory_readings = []
        
        for i in range(10):
            profile = profiling_service.create_profile(f"leak_test_{i}", adapter, source_config)
            current_memory = process.memory_info().rss
            memory_readings.append(current_memory)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        
        # Assert no significant memory leak
        assert memory_increase_mb < 100  # Should not increase by more than 100MB
        
        # Check that memory doesn't continuously grow
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        memory_trend_mb_per_iter = memory_trend / 1024 / 1024
        assert memory_trend_mb_per_iter < 5  # Should not grow by more than 5MB per iteration

    @pytest.mark.slow
    def test_memory_cleanup_after_large_operations(self, mock_data_profile_repository, tmp_path):
        """Test that memory is properly cleaned up after large operations."""
        import gc
        
        profiling_service = DataProfilingService(mock_data_profile_repository)
        adapter = PandasCSVAdapter()
        
        # Create large dataset
        size = 50000
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'text_data': [f'Large text content for record {i} ' * 20 for i in range(1, size + 1)],
            'numeric_data': np.random.uniform(0, 1000, size)
        })
        
        file_path = tmp_path / "large_cleanup_test.csv"
        data.to_csv(file_path, index=False)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform large operation
        source_config = {"file_path": str(file_path)}
        profile = profiling_service.create_profile("large_cleanup_test", adapter, source_config)
        
        peak_memory = process.memory_info().rss
        
        # Clear references and force garbage collection
        del profile
        del data
        gc.collect()
        
        # Wait a bit for memory cleanup
        time.sleep(1)
        final_memory = process.memory_info().rss
        
        # Calculate memory usage
        peak_increase_mb = (peak_memory - initial_memory) / 1024 / 1024
        final_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        cleanup_ratio = final_increase_mb / peak_increase_mb if peak_increase_mb > 0 else 0
        
        # Assert proper cleanup
        assert cleanup_ratio < 0.3  # Should cleanup at least 70% of peak memory usage