"""Tests for performance optimizations in data quality validation.

Tests for optimized validation engine, performance profiling, and resource monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import time

# Import the optimization components
from src.packages.data_quality.application.services.optimized_validation_engine import (
    OptimizedValidationEngine, OptimizedValidationConfig, MemoryMonitor
)
from src.packages.data_quality.application.services.performance_optimizer import (
    PerformanceProfiler, PerformanceProfile, ValidationCache, 
    AdaptiveExecutionStrategy, ResourceMonitor
)

# Import domain entities
from src.packages.data_quality.domain.entities.validation_rule import (
    QualityRule, ValidationLogic, ValidationResult, ValidationError,
    RuleId, RuleType, LogicType, Severity, QualityCategory,
    SuccessCriteria, UserId, ValidationStatus
)
from src.packages.data_quality.domain.entities.quality_profile import DatasetId


class TestOptimizedValidationEngine:
    """Test suite for optimized validation engine."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for testing."""
        np.random.seed(42)
        size = 50000
        
        data = {
            'id': range(1, size + 1),
            'name': [f'Name_{i}' for i in range(1, size + 1)],
            'value': np.random.randn(size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'date': pd.date_range('2020-01-01', periods=size, freq='H'),
            'is_active': np.random.choice([True, False], size),
            'score': np.random.uniform(0, 100, size)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def performance_rules(self):
        """Create rules for performance testing."""
        return [
            QualityRule(
                rule_id=RuleId(),
                rule_name="ID Uniqueness",
                description="Check ID uniqueness",
                rule_type=RuleType.UNIQUENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="~df['id'].duplicated()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID must be unique"
                ),
                target_columns=['id'],
                severity=Severity.CRITICAL,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Value Range",
                description="Check value range",
                rule_type=RuleType.RANGE,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.STATISTICAL,
                    expression="range_check",
                    parameters={'column_name': 'value', 'stat_type': 'z_score', 'threshold': 3.0},
                    success_criteria=SuccessCriteria(min_pass_rate=0.95),
                    error_message="Value is an outlier"
                ),
                target_columns=['value'],
                severity=Severity.MEDIUM,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Category Lookup",
                description="Check category values",
                rule_type=RuleType.VALIDITY,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.LOOKUP,
                    expression="category_check",
                    parameters={'column_name': 'category', 'lookup_values': ['A', 'B', 'C', 'D']},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="Invalid category value"
                ),
                target_columns=['category'],
                severity=Severity.HIGH,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
    
    @pytest.fixture
    def optimized_config(self):
        """Create optimized validation configuration."""
        return OptimizedValidationConfig(
            enable_parallel_processing=True,
            max_workers=4,
            memory_limit_mb=1024,
            chunk_size=5000,
            adaptive_chunking=True,
            enable_streaming=True,
            enable_vectorization=True,
            enable_adaptive_execution=True
        )
    
    def test_optimized_engine_initialization(self, optimized_config):
        """Test optimized engine initialization."""
        engine = OptimizedValidationEngine(optimized_config)
        
        assert engine.config == optimized_config
        assert engine.memory_monitor is not None
        assert engine.stream_processor is not None
        assert isinstance(engine.performance_metrics, dict)
        assert engine.performance_metrics['total_records_processed'] == 0
    
    def test_dataset_analysis(self, large_dataset, optimized_config):
        """Test dataset analysis functionality."""
        engine = OptimizedValidationEngine(optimized_config)
        
        dataset_info = engine._analyze_dataset(large_dataset)
        
        assert dataset_info['record_count'] == len(large_dataset)
        assert dataset_info['column_count'] == len(large_dataset.columns)
        assert dataset_info['memory_usage_mb'] > 0
        assert 'data_types' in dataset_info
        assert 'null_percentages' in dataset_info
        assert 'estimated_processing_time' in dataset_info
    
    def test_execution_strategy_selection(self, large_dataset, performance_rules, optimized_config):
        """Test execution strategy selection."""
        engine = OptimizedValidationEngine(optimized_config)
        
        # Test with different dataset sizes
        small_df = large_dataset.head(1000)
        medium_df = large_dataset.head(10000)
        large_df = large_dataset
        
        small_info = engine._analyze_dataset(small_df)
        medium_info = engine._analyze_dataset(medium_df)
        large_info = engine._analyze_dataset(large_df)
        
        small_strategy = engine._choose_execution_strategy(small_df, performance_rules, small_info)
        medium_strategy = engine._choose_execution_strategy(medium_df, performance_rules, medium_info)
        large_strategy = engine._choose_execution_strategy(large_df, performance_rules, large_info)
        
        # Verify strategy selection makes sense
        assert small_strategy in ['standard', 'chunked']
        assert medium_strategy in ['standard', 'chunked', 'streaming']
        assert large_strategy in ['chunked', 'streaming']
    
    def test_chunked_validation(self, large_dataset, performance_rules, optimized_config):
        """Test chunked validation execution."""
        engine = OptimizedValidationEngine(optimized_config)
        dataset_id = DatasetId("chunked_test")
        
        # Execute chunked validation
        results = engine._execute_chunked_validation(large_dataset, performance_rules, dataset_id)
        
        assert len(results) == len(performance_rules)
        for result in results:
            assert result.total_records == len(large_dataset)
            assert result.passed_records + result.failed_records == len(large_dataset)
    
    def test_streaming_validation(self, large_dataset, performance_rules, optimized_config):
        """Test streaming validation execution."""
        engine = OptimizedValidationEngine(optimized_config)
        dataset_id = DatasetId("streaming_test")
        
        # Execute streaming validation
        results = engine._execute_streaming_validation(large_dataset, performance_rules, dataset_id)
        
        assert len(results) == len(performance_rules)
        for result in results:
            assert result.total_records == len(large_dataset)
            assert isinstance(result.validation_id, type(results[0].validation_id))
    
    def test_memory_monitoring(self, optimized_config):
        """Test memory monitoring functionality."""
        monitor = MemoryMonitor(memory_limit_mb=512)
        
        # Test initial usage
        initial_usage = monitor.get_current_usage()
        assert initial_usage >= 0
        
        # Test peak usage tracking
        peak_usage = monitor.get_peak_usage()
        assert peak_usage >= initial_usage
        
        # Test reset
        monitor.reset_peak()
        assert monitor.get_peak_usage() == 0
    
    def test_chunk_size_optimization(self, large_dataset, optimized_config):
        """Test adaptive chunk size calculation."""
        engine = OptimizedValidationEngine(optimized_config)
        
        # Test adaptive chunking
        optimal_size = engine._calculate_optimal_chunk_size(large_dataset)
        
        assert optimal_size > 0
        assert optimal_size <= len(large_dataset)
        assert optimal_size >= 1000  # Minimum chunk size
        assert optimal_size <= 100000  # Maximum chunk size
    
    def test_data_type_optimization(self, large_dataset, optimized_config):
        """Test data type optimization."""
        engine = OptimizedValidationEngine(optimized_config)
        
        # Create test data with suboptimal types
        test_df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],
            'category_col': ['A', 'B', 'A', 'B', 'A'],
            'large_int': [1000000, 2000000, 3000000, 4000000, 5000000]
        })
        
        optimized_df = engine._optimize_data_types(test_df)
        
        # Check that optimizations were applied
        assert optimized_df['small_int'].dtype in ['uint8', 'uint16', 'uint32']
        assert optimized_df['category_col'].dtype == 'category'
    
    def test_performance_metrics_collection(self, large_dataset, performance_rules, optimized_config):
        """Test performance metrics collection."""
        engine = OptimizedValidationEngine(optimized_config)
        dataset_id = DatasetId("metrics_test")
        
        # Execute validation
        results = engine.validate_large_dataset(large_dataset, performance_rules, dataset_id)
        
        # Check metrics
        metrics = engine.get_performance_metrics()
        
        assert metrics['total_records_processed'] == len(large_dataset)
        assert metrics['total_execution_time'] > 0
        assert 'current_memory_usage' in metrics
        assert 'system_cpu_count' in metrics
        assert 'configured_workers' in metrics
    
    def test_column_pruning_optimization(self, optimized_config):
        """Test column pruning optimization."""
        engine = OptimizedValidationEngine(optimized_config)
        
        # Create DataFrame with many columns
        df = pd.DataFrame({
            'needed_col1': [1, 2, 3],
            'needed_col2': [4, 5, 6],
            'unneeded_col1': [7, 8, 9],
            'unneeded_col2': [10, 11, 12]
        })
        
        # Create rule that only needs some columns
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="Test Rule",
            description="Test rule",
            rule_type=RuleType.COMPLETENESS,
            validation_logic=ValidationLogic(
                logic_type=LogicType.EXPRESSION,
                expression="df['needed_col1'].notna()",
                parameters={'column_name': 'needed_col1'},
                success_criteria=SuccessCriteria(min_pass_rate=1.0),
                error_message="Column should not be null"
            ),
            target_columns=['needed_col1'],
            severity=Severity.HIGH,
            category=QualityCategory.DATA_INTEGRITY,
            created_by=UserId("test_user"),
            is_active=True
        )
        
        # Test optimization
        optimized_chunk = engine._optimize_chunk(df, [rule])
        
        # Should only keep needed columns when pruning is enabled
        if engine.config.enable_column_pruning:
            assert 'needed_col1' in optimized_chunk.columns
            # May keep other columns if they're referenced elsewhere
        else:
            assert len(optimized_chunk.columns) == len(df.columns)


class TestPerformanceProfiler:
    """Test suite for performance profiler."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler."""
        return PerformanceProfiler(enable_detailed_profiling=True)
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.enable_detailed_profiling == True
        assert profiler.profiles == []
        assert profiler.profiler is None
    
    def test_operation_profiling(self, profiler):
        """Test operation profiling decorator."""
        @profiler.profile_operation("test_operation")
        def test_function(data_size):
            # Simulate some work
            df = pd.DataFrame({'data': range(data_size)})
            return df.sum()
        
        # Execute profiled function
        result = test_function(1000)
        
        # Check profiling results
        assert len(profiler.profiles) == 1
        profile = profiler.profiles[0]
        
        assert profile.operation_name == "test_operation"
        assert profile.execution_time > 0
        assert profile.records_processed > 0
        assert profile.throughput_records_per_second > 0
    
    def test_performance_profile_efficiency_score(self):
        """Test performance profile efficiency scoring."""
        profile = PerformanceProfile(
            operation_name="test_op",
            execution_time=1.0,
            memory_usage_mb=256,
            cpu_usage_percent=50,
            records_processed=1000,
            throughput_records_per_second=1000
        )
        
        efficiency = profile.get_efficiency_score()
        assert 0.0 <= efficiency <= 1.0
        
        # Test with high resource usage
        high_usage_profile = PerformanceProfile(
            operation_name="high_usage_op",
            execution_time=1.0,
            memory_usage_mb=2048,  # High memory
            cpu_usage_percent=95,  # High CPU
            records_processed=1000,
            throughput_records_per_second=1000
        )
        
        high_usage_efficiency = high_usage_profile.get_efficiency_score()
        assert high_usage_efficiency < efficiency
    
    def test_optimization_opportunity_analysis(self, profiler):
        """Test optimization opportunity analysis."""
        # Create profile with various issues
        profile = PerformanceProfile(
            operation_name="slow_operation",
            execution_time=120.0,  # Long execution time
            memory_usage_mb=1500,  # High memory usage
            cpu_usage_percent=95,  # High CPU usage
            records_processed=1000,
            throughput_records_per_second=8  # Low throughput
        )
        
        profiler.profiles = [profile]
        profiler._analyze_optimization_opportunities(profile)
        
        # Check that opportunities were identified
        assert len(profile.optimization_opportunities) > 0
        assert len(profile.bottlenecks) > 0
    
    def test_performance_summary(self, profiler):
        """Test performance summary generation."""
        # Add some test profiles
        for i in range(3):
            profile = PerformanceProfile(
                operation_name=f"operation_{i}",
                execution_time=i + 1.0,
                memory_usage_mb=100 + i * 50,
                cpu_usage_percent=50 + i * 10,
                records_processed=1000 + i * 100,
                throughput_records_per_second=500 + i * 50
            )
            profiler.profiles.append(profile)
        
        summary = profiler.get_performance_summary()
        
        assert summary['total_operations'] == 3
        assert summary['total_execution_time'] > 0
        assert summary['total_memory_usage_mb'] > 0
        assert summary['total_records_processed'] > 0
        assert summary['average_throughput'] > 0
        assert summary['average_efficiency'] > 0
        assert 'optimization_opportunities' in summary
        assert 'bottlenecks' in summary


class TestValidationCache:
    """Test suite for validation cache."""
    
    @pytest.fixture
    def cache(self):
        """Create validation cache."""
        return ValidationCache(max_size=10, ttl_minutes=1)
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 10
        assert cache.ttl_minutes == 1
        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0
    
    def test_cache_put_get(self, cache):
        """Test cache put and get operations."""
        # Test put
        cache.put("key1", "value1")
        assert len(cache.cache) == 1
        
        # Test get
        value = cache.get("key1")
        assert value == "value1"
        
        # Test non-existent key
        value = cache.get("non_existent")
        assert value is None
    
    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration."""
        # Put value
        cache.put("key1", "value1")
        
        # Manually expire by setting old timestamp
        cache.cache["key1"]["created_at"] = datetime.now() - timedelta(minutes=2)
        
        # Should return None due to expiration
        value = cache.get("key1")
        assert value is None
        assert "key1" not in cache.cache
    
    def test_cache_size_limit(self, cache):
        """Test cache size limit enforcement."""
        # Fill cache to capacity
        for i in range(cache.max_size + 5):
            cache.put(f"key{i}", f"value{i}")
        
        # Should not exceed max size
        assert len(cache.cache) <= cache.max_size
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Add some entries
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['ttl_minutes'] == 1
        assert stats['oldest_entry'] is not None
        assert stats['newest_entry'] is not None
    
    def test_cache_clear(self, cache):
        """Test cache clearing."""
        # Add entries
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Clear cache
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0


class TestAdaptiveExecutionStrategy:
    """Test suite for adaptive execution strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create adaptive execution strategy."""
        return AdaptiveExecutionStrategy()
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.execution_history == []
        assert strategy.strategy_performance == {}
    
    def test_dataset_categorization(self, strategy):
        """Test dataset categorization."""
        assert strategy._categorize_dataset(5000, 5) == 'small'
        assert strategy._categorize_dataset(50000, 10) == 'medium'
        assert strategy._categorize_dataset(500000, 20) == 'large'
        assert strategy._categorize_dataset(5000000, 50) == 'xlarge'
    
    def test_strategy_selection(self, strategy):
        """Test strategy selection."""
        # Test with different dataset sizes
        small_strategy = strategy.choose_strategy(5000, 5, 1024, 4)
        medium_strategy = strategy.choose_strategy(50000, 10, 1024, 4)
        large_strategy = strategy.choose_strategy(500000, 20, 1024, 4)
        
        assert small_strategy in ['standard', 'parallel', 'chunked', 'streaming']
        assert medium_strategy in ['standard', 'parallel', 'chunked', 'streaming']
        assert large_strategy in ['standard', 'parallel', 'chunked', 'streaming']
    
    def test_strategy_feasibility_check(self, strategy):
        """Test strategy feasibility checking."""
        # Test with limited memory
        feasible = strategy._is_strategy_feasible('standard', 1000000, 128, 4)
        assert isinstance(feasible, bool)
        
        # Test with insufficient workers
        feasible = strategy._is_strategy_feasible('parallel', 10000, 1024, 1)
        assert not feasible
    
    def test_execution_recording(self, strategy):
        """Test execution recording."""
        # Record some executions
        strategy.record_execution('standard', 10000, 5, 2.5, 256)
        strategy.record_execution('parallel', 10000, 5, 1.8, 320)
        strategy.record_execution('chunked', 10000, 5, 3.2, 180)
        
        assert len(strategy.execution_history) == 3
        assert 'medium' in strategy.strategy_performance
        assert len(strategy.strategy_performance['medium']) == 3
    
    def test_performance_score_calculation(self, strategy):
        """Test performance score calculation."""
        score1 = strategy._calculate_performance_score(2.0, 256, 1000)
        score2 = strategy._calculate_performance_score(1.0, 256, 1000)  # Faster
        score3 = strategy._calculate_performance_score(2.0, 128, 1000)  # Less memory
        
        assert score2 > score1  # Faster execution should have higher score
        assert score3 > score1  # Less memory usage should have higher score
    
    def test_strategy_recommendations(self, strategy):
        """Test strategy recommendations."""
        # Record some performance data
        strategy.record_execution('standard', 10000, 5, 2.5, 256)
        strategy.record_execution('parallel', 10000, 5, 1.8, 320)
        strategy.record_execution('chunked', 10000, 5, 3.2, 180)
        
        recommendations = strategy.get_strategy_recommendations()
        
        assert len(recommendations) > 0
        for rec in recommendations:
            assert 'dataset_category' in rec
            assert 'best_strategy' in rec
            assert 'best_score' in rec
            assert 'worst_strategy' in rec
            assert 'worst_score' in rec
            assert 'improvement_potential' in rec


class TestResourceMonitor:
    """Test suite for resource monitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create resource monitor."""
        return ResourceMonitor(monitoring_interval=0.1)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.monitoring_interval == 0.1
        assert monitor.monitoring == False
        assert monitor.metrics == []
        assert monitor.monitor_thread is None
    
    def test_monitoring_start_stop(self, monitor):
        """Test monitoring start and stop."""
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring == True
        assert monitor.monitor_thread is not None
        
        # Wait a bit for metrics collection
        time.sleep(0.5)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring == False
        
        # Should have collected some metrics
        assert len(monitor.metrics) > 0
    
    def test_resource_summary(self, monitor):
        """Test resource summary generation."""
        # Add some mock metrics
        monitor.metrics = [
            {
                'timestamp': datetime.now(),
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'memory_available_mb': 1000,
                'disk_percent': 30.0,
                'disk_free_mb': 5000
            },
            {
                'timestamp': datetime.now(),
                'cpu_percent': 70.0,
                'memory_percent': 80.0,
                'memory_available_mb': 800,
                'disk_percent': 35.0,
                'disk_free_mb': 4500
            }
        ]
        
        summary = monitor.get_resource_summary()
        
        assert 'monitoring_duration_seconds' in summary
        assert 'cpu_usage' in summary
        assert 'memory_usage' in summary
        assert 'peak_memory_usage' in summary
        assert 'peak_cpu_usage' in summary
        assert 'resource_warnings' in summary
        
        # Check that averages are calculated
        assert summary['cpu_usage']['average'] == 60.0
        assert summary['memory_usage']['average'] == 70.0
    
    def test_resource_warnings(self, monitor):
        """Test resource warning generation."""
        # Add metrics with high resource usage
        monitor.metrics = [
            {
                'timestamp': datetime.now(),
                'cpu_percent': 95.0,  # High CPU
                'memory_percent': 90.0,  # High memory
                'memory_available_mb': 100,
                'disk_percent': 50.0,
                'disk_free_mb': 1000
            }
        ]
        
        warnings = monitor._get_resource_warnings()
        
        assert len(warnings) > 0
        assert any('CPU' in warning for warning in warnings)
        assert any('memory' in warning for warning in warnings)


class TestPerformanceIntegration:
    """Integration tests for performance optimizations."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create test data
        df = pd.DataFrame({
            'id': range(1, 10001),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Create rules
        rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Test Rule",
                description="Test rule",
                rule_type=RuleType.COMPLETENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="df['id'].notna()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID should not be null"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
        
        # Create optimized configuration
        config = OptimizedValidationConfig(
            enable_parallel_processing=True,
            max_workers=2,
            memory_limit_mb=512,
            chunk_size=2000,
            enable_vectorization=True
        )
        
        # Create optimized engine
        engine = OptimizedValidationEngine(config)
        
        # Create profiler
        profiler = PerformanceProfiler()
        
        # Create adaptive strategy
        strategy = AdaptiveExecutionStrategy()
        
        # Execute with profiling
        dataset_id = DatasetId("integration_test")
        
        start_time = time.time()
        results = engine.validate_large_dataset(df, rules, dataset_id)
        execution_time = time.time() - start_time
        
        # Record execution for adaptive strategy
        strategy.record_execution('optimized', len(df), len(rules), execution_time, 256)
        
        # Get performance metrics
        engine_metrics = engine.get_performance_metrics()
        strategy_recommendations = strategy.get_strategy_recommendations()
        
        # Validate results
        assert len(results) == len(rules)
        assert engine_metrics['total_records_processed'] == len(df)
        assert engine_metrics['total_execution_time'] > 0
        
        # Validate performance was reasonable
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert engine_metrics['total_records_processed'] > 0
        
        # Test that we can get recommendations
        assert isinstance(strategy_recommendations, list)
    
    def test_memory_efficiency_validation(self):
        """Test memory efficiency during validation."""
        # Create memory-intensive dataset
        df = pd.DataFrame({
            'id': range(1, 20001),
            'large_text': [f'text_data_{i}' * 100 for i in range(1, 20001)]
        })
        
        # Create configuration with memory limits
        config = OptimizedValidationConfig(
            memory_limit_mb=256,
            chunk_size=1000,
            adaptive_chunking=True,
            enable_memory_monitoring=True
        )
        
        # Create engine with memory monitoring
        engine = OptimizedValidationEngine(config)
        
        # Create simple rule
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="Memory Test Rule",
            description="Test memory efficiency",
            rule_type=RuleType.COMPLETENESS,
            validation_logic=ValidationLogic(
                logic_type=LogicType.EXPRESSION,
                expression="df['id'].notna()",
                parameters={'column_name': 'id'},
                success_criteria=SuccessCriteria(min_pass_rate=1.0),
                error_message="ID should not be null"
            ),
            target_columns=['id'],
            severity=Severity.HIGH,
            category=QualityCategory.DATA_INTEGRITY,
            created_by=UserId("test_user"),
            is_active=True
        )
        
        # Execute validation
        dataset_id = DatasetId("memory_test")
        results = engine.validate_large_dataset(df, [rule], dataset_id)
        
        # Check that validation completed successfully
        assert len(results) == 1
        assert results[0].total_records == len(df)
        
        # Check memory usage was monitored
        metrics = engine.get_performance_metrics()
        assert 'current_memory_usage' in metrics
        
        # Memory monitor should have tracked usage
        peak_usage = engine.memory_monitor.get_peak_usage()
        assert peak_usage >= 0