"""Tests for Batch Configuration Manager."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pynomaly.application.services.batch_configuration_manager import (
    BatchConfigurationManager, SystemResources, DataCharacteristics,
    ProcessingProfile, BatchOptimizationResult
)


@pytest.fixture
def config_manager():
    """Create a batch configuration manager for testing."""
    return BatchConfigurationManager()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col': range(1000),
        'float_col': [x * 1.5 for x in range(1000)],
        'text_col': [f'text_{x}' for x in range(1000)],
        'category_col': [f'cat_{x % 5}' for x in range(1000)]
    })


class TestSystemResources:
    """Test system resource monitoring."""
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_capture_current(self, mock_pids, mock_net, mock_disk, mock_memory, mock_cpu_percent, mock_cpu_count):
        """Test capturing current system resources."""
        # Mock return values
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 25.5
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 16 * 1024**3  # 16GB
        mock_memory_obj.available = 8 * 1024**3  # 8GB
        mock_memory_obj.percent = 50.0
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = MagicMock()
        mock_disk_obj.percent = 75.0
        mock_disk.return_value = mock_disk_obj
        
        mock_net_obj = MagicMock()
        mock_net_obj.bytes_sent = 1000000
        mock_net_obj.bytes_recv = 2000000
        mock_net.return_value = mock_net_obj
        
        mock_pids.return_value = list(range(200))
        
        resources = SystemResources.get_current()
        
        assert resources.cpu_count == 8
        assert resources.cpu_percent == 25.5
        assert resources.memory_percent == 50.0
        assert resources.memory_available_gb == 8.0
        assert resources.disk_usage_percent == 75.0
        assert resources.network_bytes_sent == 1000000
        assert resources.network_bytes_recv == 2000000
        assert resources.process_count == 200


class TestDataCharacteristics:
    """Test data characteristics analysis."""
    
    def test_analyze_dataframe_basic(self, sample_dataframe):
        """Test basic DataFrame analysis."""
        characteristics = DataCharacteristics.analyze_dataframe(sample_dataframe)
        
        assert characteristics.total_rows == 1000
        assert characteristics.total_columns == 4
        assert characteristics.memory_usage_mb > 0
        assert characteristics.has_numeric_data is True
        assert characteristics.has_text_data is True
        assert characteristics.average_row_size_bytes > 0
        assert 0 <= characteristics.complexity_score <= 1
    
    def test_analyze_dataframe_data_types(self):
        """Test data type analysis."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'cat_col': pd.Categorical(['x', 'y', 'z'])
        })
        
        characteristics = DataCharacteristics.analyze_dataframe(df)
        
        assert characteristics.has_numeric_data is True
        assert characteristics.has_text_data is True
        assert characteristics.has_categorical_data is True
        assert len(characteristics.data_types) == 4
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        # Simple DataFrame (low complexity)
        simple_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        simple_score = DataCharacteristics._calculate_complexity_score(simple_df)
        
        # Complex DataFrame (high complexity)
        complex_df = pd.DataFrame({
            f'col_{i}': [f'text_{j}' if i % 2 == 0 else j for j in range(100)]
            for i in range(50)
        })
        # Add missing values
        complex_df.iloc[::10, ::5] = None
        complex_score = DataCharacteristics._calculate_complexity_score(complex_df)
        
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
        assert complex_score > simple_score


class TestProcessingProfile:
    """Test processing profile functionality."""
    
    def test_get_default_profiles(self):
        """Test getting default processing profiles."""
        profiles = ProcessingProfile.get_default_profiles()
        
        assert 'anomaly_detection' in profiles
        assert 'data_quality' in profiles
        assert 'data_profiling' in profiles
        assert 'feature_engineering' in profiles
        assert 'model_training' in profiles
        assert 'data_export' in profiles
        
        # Test specific profile properties
        anomaly_profile = profiles['anomaly_detection']
        assert anomaly_profile.cpu_intensive is True
        assert anomaly_profile.memory_intensive is True
        assert anomaly_profile.estimated_processing_time_per_row_ms > 0
        
        export_profile = profiles['data_export']
        assert export_profile.io_intensive is True
        assert export_profile.network_intensive is True


class TestBatchConfigurationManager:
    """Test batch configuration manager functionality."""
    
    def test_initialization(self, config_manager):
        """Test manager initialization."""
        assert config_manager.settings is not None
        assert len(config_manager.processing_profiles) > 0
        assert config_manager.min_batch_size > 0
        assert config_manager.max_batch_size > config_manager.min_batch_size
        assert config_manager.memory_safety_margin > 0
    
    def test_register_processing_profile(self, config_manager):
        """Test registering custom processing profiles."""
        custom_profile = ProcessingProfile(
            processor_name="custom_processor",
            cpu_intensive=True,
            memory_intensive=False,
            estimated_processing_time_per_row_ms=5.0
        )
        
        config_manager.register_processing_profile(custom_profile)
        
        assert "custom_processor" in config_manager.processing_profiles
        assert config_manager.processing_profiles["custom_processor"] == custom_profile
    
    def test_analyze_data_dataframe(self, config_manager, sample_dataframe):
        """Test data analysis for DataFrame."""
        characteristics = config_manager._analyze_data(sample_dataframe)
        
        assert characteristics.total_rows == 1000
        assert characteristics.total_columns == 4
        assert characteristics.memory_usage_mb > 0
        assert characteristics.has_numeric_data is True
    
    def test_analyze_data_list(self, config_manager):
        """Test data analysis for list."""
        data = list(range(100))
        characteristics = config_manager._analyze_data(data)
        
        assert characteristics.total_rows == 100
        assert characteristics.total_columns == 1
        assert characteristics.memory_usage_mb > 0
    
    def test_analyze_data_unknown_type(self, config_manager):
        """Test data analysis for unknown data type."""
        data = "single_string"
        characteristics = config_manager._analyze_data(data)
        
        assert characteristics.total_rows == 1
        assert characteristics.total_columns == 1
        assert characteristics.complexity_score == 0.1
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_calculate_optimal_batch_config(self, mock_memory, mock_cpu_percent, mock_cpu_count, 
                                          config_manager, sample_dataframe):
        """Test optimal batch configuration calculation."""
        # Mock system resources
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 30.0
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 16 * 1024**3  # 16GB
        mock_memory_obj.available = 8 * 1024**3  # 8GB
        mock_memory_obj.percent = 50.0
        mock_memory.return_value = mock_memory_obj
        
        result = config_manager.calculate_optimal_batch_config(
            data=sample_dataframe,
            processor_name="anomaly_detection"
        )
        
        assert isinstance(result, BatchOptimizationResult)
        assert result.recommended_batch_size >= config_manager.min_batch_size
        assert result.recommended_batch_size <= config_manager.max_batch_size
        assert result.recommended_concurrency >= config_manager.min_concurrency
        assert result.estimated_memory_usage_mb > 0
        assert result.estimated_processing_time_seconds > 0
        assert 0 <= result.confidence_score <= 1
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_calculate_optimal_config_with_constraints(self, mock_memory, mock_cpu_percent, mock_cpu_count,
                                                     config_manager, sample_dataframe):
        """Test optimal configuration with constraints."""
        # Mock system resources
        mock_cpu_count.return_value = 4
        mock_cpu_percent.return_value = 80.0  # High CPU usage
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 4 * 1024**3  # 4GB
        mock_memory_obj.available = 1 * 1024**3  # 1GB available
        mock_memory_obj.percent = 75.0  # High memory usage
        mock_memory.return_value = mock_memory_obj
        
        result = config_manager.calculate_optimal_batch_config(
            data=sample_dataframe,
            processor_name="anomaly_detection",
            target_memory_usage_mb=500.0,
            max_concurrency=2
        )
        
        # Should recommend smaller batch sizes due to constraints
        assert result.recommended_batch_size <= 1000  # Reasonable for constrained system
        assert result.recommended_concurrency <= 2  # Respects max_concurrency
        assert result.estimated_memory_usage_mb <= 1000  # Should fit in available memory
        assert len(result.warnings) > 0  # Should have warnings about resource constraints
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent') 
    @patch('psutil.virtual_memory')
    def test_calculate_optimal_config_sequential_processor(self, mock_memory, mock_cpu_percent, mock_cpu_count,
                                                         config_manager, sample_dataframe):
        """Test configuration for processor requiring sequential processing."""
        # Mock system resources
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 20.0
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 16 * 1024**3
        mock_memory_obj.available = 12 * 1024**3
        mock_memory_obj.percent = 25.0
        mock_memory.return_value = mock_memory_obj
        
        # Register sequential processor
        sequential_profile = ProcessingProfile(
            processor_name="sequential_processor",
            requires_order=True,
            cpu_intensive=True
        )
        config_manager.register_processing_profile(sequential_profile)
        
        result = config_manager.calculate_optimal_batch_config(
            data=sample_dataframe,
            processor_name="sequential_processor"
        )
        
        assert result.recommended_concurrency == 1  # Must be 1 for sequential
        assert any("Sequential processing required" in warning for warning in result.warnings)
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_create_optimized_config(self, mock_memory, mock_cpu_percent, mock_cpu_count,
                                   config_manager, sample_dataframe):
        """Test creating optimized BatchConfig."""
        # Mock system resources
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 25.0
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 16 * 1024**3
        mock_memory_obj.available = 10 * 1024**3
        mock_memory_obj.percent = 37.5
        mock_memory.return_value = mock_memory_obj
        
        config = config_manager.create_optimized_config(
            data=sample_dataframe,
            processor_name="data_quality",
            timeout_seconds=120,
            retry_attempts=5
        )
        
        from src.pynomaly.application.services.batch_processing_service import BatchConfig
        assert isinstance(config, BatchConfig)
        assert config.batch_size >= config_manager.min_batch_size
        assert config.max_concurrent_batches >= 1
        assert config.memory_limit_mb > 0
        assert config.timeout_seconds == 120  # Should preserve override
        assert config.retry_attempts == 5  # Should preserve override
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_system_recommendations(self, mock_memory, mock_cpu_percent, mock_cpu_count, config_manager):
        """Test getting system recommendations."""
        # Mock high resource usage scenario
        mock_cpu_count.return_value = 4
        mock_cpu_percent.return_value = 85.0  # High CPU
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 8 * 1024**3
        mock_memory_obj.available = 0.3 * 1024**3  # Low available memory
        mock_memory_obj.percent = 90.0  # High memory usage
        mock_memory.return_value = mock_memory_obj
        
        recommendations = config_manager.get_system_recommendations()
        
        assert "system_status" in recommendations
        assert "recommendations" in recommendations
        
        system_status = recommendations["system_status"]
        assert system_status["cpu_usage"] == 85.0
        assert system_status["memory_usage"] == 90.0
        assert system_status["cpu_cores"] == 4
        
        # Should have warnings about high resource usage
        rec_messages = [r["message"] for r in recommendations["recommendations"]]
        assert any("High CPU usage" in msg for msg in rec_messages)
        assert any("High memory usage" in msg or "Low available memory" in msg for msg in rec_messages)
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_system_recommendations_healthy(self, mock_memory, mock_cpu_percent, mock_cpu_count, config_manager):
        """Test system recommendations with healthy resources."""
        # Mock healthy system
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 25.0  # Low CPU
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.total = 16 * 1024**3
        mock_memory_obj.available = 12 * 1024**3  # Plenty available
        mock_memory_obj.percent = 25.0  # Low memory usage
        mock_memory.return_value = mock_memory_obj
        
        recommendations = config_manager.get_system_recommendations()
        
        # Should suggest optimization opportunities
        rec_messages = [r["message"] for r in recommendations["recommendations"]]
        assert any("consider increasing" in msg.lower() for msg in rec_messages)
        assert any("ample memory" in msg.lower() or "larger batch" in msg.lower() for msg in rec_messages)


class TestOptimizationFactors:
    """Test optimization factor calculations."""
    
    def test_memory_based_calculation(self, config_manager):
        """Test memory-based batch size calculation."""
        # Create data with known characteristics
        data = pd.DataFrame({'col': range(1000)})
        characteristics = DataCharacteristics.analyze_dataframe(data)
        
        # Should calculate reasonable batch size based on memory
        assert characteristics.average_row_size_bytes > 0
        assert characteristics.memory_usage_mb > 0
    
    def test_complexity_adjustment(self, config_manager):
        """Test complexity-based adjustments."""
        # Simple data (low complexity)
        simple_data = pd.DataFrame({'numbers': range(100)})
        simple_characteristics = DataCharacteristics.analyze_dataframe(simple_data)
        
        # Complex data (high complexity)
        complex_data = pd.DataFrame({
            f'text_col_{i}': [f'long_text_string_{j}' * 10 for j in range(100)]
            for i in range(20)
        })
        complex_characteristics = DataCharacteristics.analyze_dataframe(complex_data)
        
        assert simple_characteristics.complexity_score < complex_characteristics.complexity_score
    
    def test_optimization_warnings(self, config_manager):
        """Test warning generation in optimization."""
        # Create scenario that should generate warnings
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory_obj = MagicMock()
            mock_memory_obj.total = 2 * 1024**3  # 2GB total
            mock_memory_obj.available = 0.1 * 1024**3  # 100MB available
            mock_memory_obj.percent = 95.0
            mock_memory.return_value = mock_memory_obj
            
            large_data = pd.DataFrame({
                'col': range(10000)  # Large dataset
            })
            
            result = config_manager.calculate_optimal_batch_config(
                data=large_data,
                processor_name="anomaly_detection"
            )
            
            # Should generate warnings about memory constraints
            assert len(result.warnings) > 0
            assert any("memory" in warning.lower() for warning in result.warnings)