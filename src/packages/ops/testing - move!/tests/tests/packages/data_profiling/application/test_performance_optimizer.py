import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import multiprocessing

from src.packages.data_profiling.application.services.performance_optimizer import PerformanceOptimizer


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer service."""
    
    @pytest.fixture
    def optimizer(self):
        """Create PerformanceOptimizer instance."""
        return PerformanceOptimizer()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            'numeric_col': np.random.normal(0, 1, 1000),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000),
            'string_col': [f'value_{i}' for i in range(1000)],
            'datetime_col': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'boolean_col': np.random.choice([True, False], 1000)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def large_dataframe(self):
        """Create large DataFrame for testing sampling."""
        np.random.seed(42)
        data = {
            'col1': np.random.normal(0, 1, 50000),
            'col2': np.random.choice(['X', 'Y', 'Z'], 50000),
            'col3': range(50000)
        }
        return pd.DataFrame(data)


class TestIntelligentSampling:
    """Test intelligent sampling functionality."""
    
    def test_no_sampling_needed_small_dataset(self, optimizer, sample_dataframe):
        """Test that small datasets are not sampled."""
        result = optimizer.apply_intelligent_sampling(sample_dataframe, target_size=2000)
        
        assert len(result) == len(sample_dataframe)
        assert result.equals(sample_dataframe)
    
    def test_sampling_large_dataset(self, optimizer, large_dataframe):
        """Test sampling of large dataset."""
        target_size = 1000
        result = optimizer.apply_intelligent_sampling(large_dataframe, target_size=target_size)
        
        assert len(result) == target_size
        assert set(result.columns) == set(large_dataframe.columns)
    
    def test_percentage_sampling(self, optimizer, large_dataframe):
        """Test percentage-based sampling."""
        target_percentage = 10.0
        result = optimizer.apply_intelligent_sampling(
            large_dataframe, 
            target_percentage=target_percentage
        )
        
        expected_size = int(len(large_dataframe) * target_percentage / 100)
        assert len(result) == expected_size
    
    def test_stratified_sampling(self, optimizer, large_dataframe):
        """Test stratified sampling with stratify column."""
        result = optimizer.apply_intelligent_sampling(
            large_dataframe,
            target_size=1000,
            stratify_column='col2'
        )
        
        assert len(result) == 1000
        
        # Check that all strata are represented
        original_distribution = large_dataframe['col2'].value_counts(normalize=True)
        sampled_distribution = result['col2'].value_counts(normalize=True)
        
        # Allow some tolerance for sampling variance
        for category in original_distribution.index:
            assert abs(
                original_distribution[category] - sampled_distribution[category]
            ) < 0.1
    
    def test_systematic_sampling_time_series(self, optimizer):
        """Test systematic sampling for time series data."""
        # Create time series DataFrame
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H'),
            'value': np.random.normal(0, 1, 10000)
        }
        df = pd.DataFrame(data)
        
        result = optimizer.apply_intelligent_sampling(df, target_size=1000)
        
        assert len(result) == 1000
        # Systematic sampling should maintain temporal order
        assert result['timestamp'].is_monotonic_increasing
    
    def test_sampling_fallback_on_error(self, optimizer, large_dataframe):
        """Test fallback to random sampling on error."""
        with patch.object(optimizer, '_adaptive_sampling', side_effect=Exception("Test error")):
            result = optimizer.apply_intelligent_sampling(large_dataframe, target_size=1000)
            
            assert len(result) == 1000
            assert set(result.columns) == set(large_dataframe.columns)


class TestMemoryOptimization:
    """Test memory optimization functionality."""
    
    def test_optimize_memory_usage(self, optimizer, sample_dataframe):
        """Test memory usage optimization."""
        result = optimizer.optimize_memory_usage(sample_dataframe)
        
        # Should preserve data integrity
        assert len(result) == len(sample_dataframe)
        assert set(result.columns) == set(sample_dataframe.columns)
        
        # Memory usage should be reduced or same
        original_memory = sample_dataframe.memory_usage(deep=True).sum()
        optimized_memory = result.memory_usage(deep=True).sum()
        assert optimized_memory <= original_memory
    
    def test_numeric_downcast(self, optimizer):
        """Test numeric type downcasting."""
        # Create DataFrame with large numeric types
        data = {
            'int_col': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64')
        }
        df = pd.DataFrame(data)
        
        result = optimizer.optimize_memory_usage(df)
        
        # Check if types were downcasted
        assert result['int_col'].dtype.name in ['int8', 'int16', 'int32']
        assert result['float_col'].dtype.name in ['float32']
    
    def test_categorical_conversion(self, optimizer):
        """Test categorical conversion for low cardinality strings."""
        data = {
            'high_cardinality': [f'unique_{i}' for i in range(1000)],  # Won't be converted
            'low_cardinality': ['A', 'B', 'C'] * 334  # Will be converted
        }
        df = pd.DataFrame(data)
        
        result = optimizer.optimize_memory_usage(df)
        
        # High cardinality should remain object
        assert result['high_cardinality'].dtype == 'object'
        
        # Low cardinality should become category
        assert result['low_cardinality'].dtype.name == 'category'


class TestDataTypeOptimization:
    """Test data type optimization functionality."""
    
    def test_optimize_data_types(self, optimizer):
        """Test data type optimization."""
        data = {
            'date_string': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'numeric_string': ['123', '456', '789'],
            'category_string': ['A', 'B', 'A', 'B', 'C'] * 200,
            'mixed_string': ['text', '123', 'more_text']
        }
        df = pd.DataFrame(data)
        
        result = optimizer.optimize_data_types(df)
        
        # Date strings should be converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date_string'])
        
        # Numeric strings should be converted to numeric
        assert pd.api.types.is_numeric_dtype(result['numeric_string'])
        
        # Category strings should become categorical
        assert result['category_string'].dtype.name == 'category'
        
        # Mixed strings should remain object
        assert result['mixed_string'].dtype == 'object'
    
    def test_datetime_detection(self, optimizer):
        """Test datetime column detection."""
        test_series = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        assert optimizer._is_datetime_column(test_series) is True
        
        test_series = pd.Series(['not', 'a', 'date'])
        assert optimizer._is_datetime_column(test_series) is False
    
    def test_numeric_detection(self, optimizer):
        """Test numeric column detection."""
        test_series = pd.Series(['123', '456', '789'])
        assert optimizer._is_numeric_column(test_series) is True
        
        test_series = pd.Series(['123', 'abc', '789'])
        assert optimizer._is_numeric_column(test_series) is False
    
    def test_category_detection(self, optimizer):
        """Test categorical column detection."""
        # Low cardinality should be categorical
        test_series = pd.Series(['A', 'B', 'C'] * 100)
        assert optimizer._should_be_category(test_series) is True
        
        # High cardinality should not be categorical
        test_series = pd.Series([f'unique_{i}' for i in range(300)])
        assert optimizer._should_be_category(test_series) is False
        
        # Too few unique values should not be categorical
        test_series = pd.Series(['A', 'B'])
        assert optimizer._should_be_category(test_series) is False


class TestParallelProcessing:
    """Test parallel processing functionality."""
    
    def test_parallelize_operation(self, optimizer, sample_dataframe):
        """Test parallel operation execution."""
        def dummy_operation(df_partition):
            """Dummy operation for testing."""
            return df_partition.copy()
        
        result = optimizer.parallelize_operation(sample_dataframe, dummy_operation, n_partitions=2)
        
        assert len(result) == len(sample_dataframe)
        assert set(result.columns) == set(sample_dataframe.columns)
    
    def test_parallel_operation_fallback(self, optimizer, sample_dataframe):
        """Test fallback to sequential processing on error."""
        def failing_operation(df_partition):
            """Operation that fails in parallel but works sequentially."""
            # This will fail in multiprocessing but work in main process
            if multiprocessing.current_process().name != 'MainProcess':
                raise Exception("Parallel failure")
            return df_partition.copy()
        
        # Should fallback to sequential processing
        result = optimizer.parallelize_operation(sample_dataframe, failing_operation)
        
        assert len(result) == len(sample_dataframe)
    
    def test_chunk_processing(self, optimizer, sample_dataframe):
        """Test chunk-based processing."""
        def dummy_operation(df_chunk):
            """Dummy operation for testing."""
            return df_chunk.copy()
        
        result = optimizer.process_in_chunks(sample_dataframe, dummy_operation, chunk_size=100)
        
        assert len(result) == len(sample_dataframe)
        assert set(result.columns) == set(sample_dataframe.columns)


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def test_monitor_performance(self, optimizer):
        """Test performance monitoring."""
        def test_operation(x, y):
            """Simple test operation."""
            return x + y
        
        result, metrics = optimizer.monitor_performance(test_operation, 5, 10)
        
        assert result == 15
        assert 'execution_time_seconds' in metrics
        assert 'memory_usage_mb' in metrics
        assert 'cpu_usage_percent' in metrics
        assert 'peak_memory_mb' in metrics
        assert metrics['execution_time_seconds'] >= 0
    
    @patch('psutil.Process')
    def test_monitor_performance_fallback(self, mock_process, optimizer):
        """Test performance monitoring fallback on error."""
        mock_process.side_effect = Exception("Process error")
        
        def test_operation():
            return "success"
        
        result, metrics = optimizer.monitor_performance(test_operation)
        
        assert result == "success"
        assert metrics == {}


class TestOptimizationRecommendations:
    """Test optimization recommendations."""
    
    def test_get_optimization_recommendations_large_dataset(self, optimizer, large_dataframe):
        """Test recommendations for large dataset."""
        recommendations = optimizer.get_optimization_recommendations(large_dataframe)
        
        assert isinstance(recommendations, list)
        assert any('sampling' in rec.lower() for rec in recommendations)
        assert any('parallel' in rec.lower() for rec in recommendations)
    
    def test_get_optimization_recommendations_small_dataset(self, optimizer, sample_dataframe):
        """Test recommendations for small dataset."""
        recommendations = optimizer.get_optimization_recommendations(sample_dataframe)
        
        assert isinstance(recommendations, list)
        # Small dataset should have fewer recommendations
    
    def test_estimate_processing_time(self, optimizer, sample_dataframe):
        """Test processing time estimation."""
        time_estimate = optimizer.estimate_processing_time(sample_dataframe, "profiling")
        
        assert isinstance(time_estimate, float)
        assert time_estimate > 0
    
    def test_estimate_processing_time_different_operations(self, optimizer, sample_dataframe):
        """Test time estimation for different operation types."""
        profiling_time = optimizer.estimate_processing_time(sample_dataframe, "profiling")
        sampling_time = optimizer.estimate_processing_time(sample_dataframe, "sampling")
        
        # Sampling should be faster than profiling
        assert sampling_time < profiling_time


class TestResourceManagement:
    """Test resource management functionality."""
    
    def test_cleanup_resources(self, optimizer):
        """Test resource cleanup."""
        # Add some cache entries first
        optimizer.get_dataset_statistics("test_hash", (1000, 10))
        
        # Cache should have entries
        cache_info = optimizer.get_dataset_statistics.cache_info()
        assert cache_info.currsize > 0
        
        # Clean up resources
        optimizer.cleanup_resources()
        
        # Cache should be cleared
        cache_info = optimizer.get_dataset_statistics.cache_info()
        assert cache_info.currsize == 0
    
    def test_cached_statistics(self, optimizer):
        """Test cached dataset statistics."""
        stats1 = optimizer.get_dataset_statistics("test_hash", (1000, 10))
        stats2 = optimizer.get_dataset_statistics("test_hash", (1000, 10))
        
        # Should return the same cached result
        assert stats1 == stats2
        assert 'shape' in stats1
        assert 'memory_usage_mb' in stats1
        assert 'optimization_recommended' in stats1


class TestDataCharacteristics:
    """Test data characteristics detection."""
    
    def test_is_time_series_data(self, optimizer):
        """Test time series data detection."""
        # DataFrame with datetime column
        ts_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'value': np.random.normal(0, 1, 100)
        })
        
        assert optimizer._is_time_series_data(ts_data) is True
        
        # DataFrame without datetime column
        non_ts_data = pd.DataFrame({
            'col1': range(100),
            'col2': np.random.normal(0, 1, 100)
        })
        
        assert optimizer._is_time_series_data(non_ts_data) is False
    
    def test_has_high_variance(self, optimizer):
        """Test high variance detection."""
        # High variance data
        high_var_data = pd.DataFrame({
            'col1': np.random.normal(0, 100, 1000),  # High variance
            'col2': np.random.normal(0, 50, 1000)    # High variance
        })
        
        assert optimizer._has_high_variance(high_var_data) is True
        
        # Low variance data
        low_var_data = pd.DataFrame({
            'col1': np.random.normal(100, 1, 1000),  # Low CV
            'col2': np.random.normal(50, 2, 1000)    # Low CV
        })
        
        assert optimizer._has_high_variance(low_var_data) is False
    
    def test_has_high_variance_no_numeric_columns(self, optimizer):
        """Test high variance detection with no numeric columns."""
        string_data = pd.DataFrame({
            'col1': ['A', 'B', 'C'] * 100,
            'col2': ['X', 'Y', 'Z'] * 100
        })
        
        assert optimizer._has_high_variance(string_data) is False