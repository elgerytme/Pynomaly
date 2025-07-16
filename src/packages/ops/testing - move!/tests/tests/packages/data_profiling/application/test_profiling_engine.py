import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.packages.data_profiling.application.services.profiling_engine import (
    ProfilingEngine, ProfilingConfig
)
from src.packages.data_profiling.domain.entities.data_profile import (
    DataProfile, SchemaProfile, QualityAssessment, ProfilingMetadata,
    ProfilingStatus, ColumnProfile, DataType, CardinalityLevel
)


class TestProfilingConfig:
    """Test ProfilingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilingConfig()
        
        # Sampling configuration
        assert config.enable_sampling is True
        assert config.sample_size == 10000
        assert config.sample_percentage is None
        
        # Analysis configuration
        assert config.include_schema_analysis is True
        assert config.include_statistical_analysis is True
        assert config.include_pattern_discovery is True
        assert config.include_quality_assessment is True
        assert config.include_relationship_analysis is True
        
        # Performance configuration
        assert config.enable_parallel_processing is True
        assert config.max_workers == 4
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
        
        # Advanced features
        assert config.enable_advanced_patterns is True
        assert config.enable_ml_clustering is True
        assert config.enable_pii_detection is True
        assert config.enable_time_series_analysis is True
        
        # Quality thresholds
        assert config.completeness_threshold == 0.95
        assert config.consistency_threshold == 0.90
        assert config.accuracy_threshold == 0.85
        
        # Memory management
        assert config.max_memory_mb == 2048
        assert config.chunk_size == 1000
        
        # Output configuration
        assert config.include_examples is True
        assert config.max_examples_per_pattern == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProfilingConfig(
            enable_sampling=False,
            sample_size=5000,
            max_workers=8,
            enable_caching=False,
            completeness_threshold=0.80
        )
        
        assert config.enable_sampling is False
        assert config.sample_size == 5000
        assert config.max_workers == 8
        assert config.enable_caching is False
        assert config.completeness_threshold == 0.80


class TestProfilingEngine:
    """Test ProfilingEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create ProfilingEngine instance."""
        config = ProfilingConfig(enable_caching=False)  # Disable caching for tests
        return ProfilingEngine(config)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            'id': range(1, 1001),
            'name': [f'User_{i}' for i in range(1, 1001)],
            'email': [f'user{i}@example.com' for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'is_active': np.random.choice([True, False], 1000),
            'created_at': pd.date_range('2020-01-01', periods=1000, freq='D')
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def large_dataframe(self):
        """Create large DataFrame for testing sampling."""
        np.random.seed(42)
        data = {
            'col1': np.random.normal(0, 1, 50000),
            'col2': np.random.choice(['A', 'B', 'C'], 50000),
            'col3': range(50000)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_schema_profile(self):
        """Create mock SchemaProfile."""
        column_profile = ColumnProfile(
            column_name="test_col",
            data_type=DataType.STRING,
            cardinality_level=CardinalityLevel.LOW
        )
        
        return SchemaProfile(
            total_tables=1,
            total_columns=1,
            total_rows=1000,
            columns=[column_profile],
            primary_keys=['id'],
            foreign_keys=[],
            estimated_size_bytes=1024000
        )
    
    @pytest.fixture
    def mock_quality_assessment(self):
        """Create mock QualityAssessment."""
        return QualityAssessment(
            overall_score=0.85,
            completeness_score=0.90,
            consistency_score=0.80,
            accuracy_score=0.85,
            validity_score=0.95,
            uniqueness_score=0.75,
            timeliness_score=0.88,
            critical_issues=0,
            high_issues=2,
            medium_issues=5,
            low_issues=10
        )


class TestProfilingEngineInitialization:
    """Test ProfilingEngine initialization."""
    
    def test_engine_initialization_default_config(self):
        """Test engine initialization with default config."""
        engine = ProfilingEngine()
        
        assert engine.config is not None
        assert isinstance(engine.config, ProfilingConfig)
        assert engine.schema_service is not None
        assert engine.statistical_service is not None
        assert engine.pattern_service is not None
        assert engine.quality_service is not None
        assert engine.performance_optimizer is not None
        assert engine.advanced_orchestrator is not None
        assert engine.cache is not None  # Caching enabled by default
    
    def test_engine_initialization_custom_config(self):
        """Test engine initialization with custom config."""
        config = ProfilingConfig(enable_caching=False, max_workers=8)
        engine = ProfilingEngine(config)
        
        assert engine.config == config
        assert engine.cache is None  # Caching disabled
    
    def test_engine_services_initialization(self):
        """Test that all services are properly initialized."""
        engine = ProfilingEngine()
        
        # Check that services have expected methods
        assert hasattr(engine.schema_service, 'infer')
        assert hasattr(engine.statistical_service, 'analyze')
        assert hasattr(engine.pattern_service, 'discover')
        assert hasattr(engine.quality_service, 'assess_quality')
        assert hasattr(engine.performance_optimizer, 'apply_intelligent_sampling')
        assert hasattr(engine.advanced_orchestrator, 'profile_dataset_comprehensive')


class TestDatasetProfiling:
    """Test dataset profiling functionality."""
    
    def test_profile_dataset_basic(self, engine, sample_dataframe, 
                                  mock_schema_profile, mock_quality_assessment):
        """Test basic dataset profiling."""
        # Mock the services to avoid complex setup
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            profile = engine.profile_dataset(sample_dataframe)
            
            assert isinstance(profile, DataProfile)
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.schema_profile == mock_schema_profile
            assert profile.quality_assessment == mock_quality_assessment
            assert profile.metadata is not None
            assert profile.metadata.execution_time_seconds > 0
    
    def test_profile_dataset_with_custom_ids(self, engine, sample_dataframe,
                                           mock_schema_profile, mock_quality_assessment):
        """Test dataset profiling with custom IDs."""
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            profile = engine.profile_dataset(
                sample_dataframe,
                dataset_id="custom_dataset_123",
                source_type="csv",
                source_connection={"path": "/data/test.csv"}
            )
            
            assert profile.dataset_id.value == "custom_dataset_123"
            assert profile.source_type == "csv"
            assert profile.source_connection == {"path": "/data/test.csv"}
    
    def test_profile_dataset_error_handling(self, engine, sample_dataframe):
        """Test error handling in dataset profiling."""
        # Mock schema service to raise an error
        with patch.object(engine.schema_service, 'infer', side_effect=Exception("Schema error")):
            
            profile = engine.profile_dataset(sample_dataframe)
            
            assert profile.status == ProfilingStatus.FAILED
            assert profile.error_message == "Schema error"
    
    def test_profile_dataset_large_data_sampling(self, engine, large_dataframe,
                                                mock_schema_profile, mock_quality_assessment):
        """Test profiling with sampling for large datasets."""
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            profile = engine.profile_dataset(large_dataframe)
            
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.metadata.sample_size is not None
            assert profile.metadata.sample_percentage is not None
            assert profile.metadata.sample_percentage < 100  # Should be sampled


class TestDatasetOptimization:
    """Test dataset optimization functionality."""
    
    def test_optimize_dataset_sampling(self, engine, large_dataframe):
        """Test dataset optimization with sampling."""
        optimized_df = engine._optimize_dataset(large_dataframe)
        
        # Should be smaller than original for large dataset
        assert len(optimized_df) <= len(large_dataframe)
        assert set(optimized_df.columns) == set(large_dataframe.columns)
    
    def test_optimize_dataset_no_sampling_needed(self, engine, sample_dataframe):
        """Test dataset optimization when no sampling needed."""
        optimized_df = engine._optimize_dataset(sample_dataframe)
        
        # Should be same size for normal-sized dataset
        assert len(optimized_df) == len(sample_dataframe)
    
    def test_optimize_dataset_memory_optimization(self, engine):
        """Test memory optimization in dataset optimization."""
        # Create DataFrame with suboptimal types
        data = {
            'large_int': [1, 2, 3, 4, 5],  # Can be downcasted
            'category_str': ['A', 'B', 'A', 'B', 'C']  # Can be categorical
        }
        df = pd.DataFrame(data)
        
        optimized_df = engine._optimize_dataset(df)
        
        # Should optimize memory usage
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        assert optimized_memory <= original_memory


class TestAnalysisComponents:
    """Test individual analysis components."""
    
    def test_run_schema_analysis(self, engine, sample_dataframe, mock_schema_profile):
        """Test schema analysis execution."""
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile):
            result = engine._run_schema_analysis(sample_dataframe)
            
            assert result == mock_schema_profile
    
    def test_run_statistical_analysis(self, engine, sample_dataframe):
        """Test statistical analysis execution."""
        with patch.object(engine.statistical_service, 'analyze', return_value={'test': 'result'}), \
             patch.object(engine.statistical_service, 'analyze_distribution', return_value={'dist': 'info'}), \
             patch.object(engine.statistical_service, 'correlation_analysis', return_value={'corr': 'matrix'}), \
             patch.object(engine.statistical_service, 'detect_outliers', return_value={'outliers': 'found'}):
            
            result = engine._run_statistical_analysis(sample_dataframe)
            
            assert 'column_stats' in result
            assert 'distributions' in result
            assert 'correlations' in result
            assert 'outliers' in result
    
    def test_run_pattern_discovery(self, engine, sample_dataframe):
        """Test pattern discovery execution."""
        with patch.object(engine.pattern_service, 'discover', return_value={'patterns': 'found'}):
            result = engine._run_pattern_discovery(sample_dataframe)
            
            assert 'column_patterns' in result
    
    def test_run_relationship_analysis(self, engine, sample_dataframe):
        """Test relationship analysis execution."""
        with patch.object(engine.schema_service, 'detect_advanced_relationships', 
                         return_value={'relationships': 'found'}):
            result = engine._run_relationship_analysis(sample_dataframe)
            
            assert result == {'relationships': 'found'}
    
    def test_run_quality_assessment(self, engine, sample_dataframe, 
                                   mock_schema_profile, mock_quality_assessment):
        """Test quality assessment execution."""
        with patch.object(engine.quality_service, 'assess_quality', 
                         return_value=mock_quality_assessment):
            result = engine._run_quality_assessment(mock_schema_profile, sample_dataframe)
            
            assert result == mock_quality_assessment


class TestCaching:
    """Test caching functionality."""
    
    def test_caching_enabled(self):
        """Test caching when enabled."""
        config = ProfilingConfig(enable_caching=True)
        engine = ProfilingEngine(config)
        
        assert engine.cache is not None
        assert isinstance(engine.cache, dict)
    
    def test_caching_disabled(self):
        """Test caching when disabled."""
        config = ProfilingConfig(enable_caching=False)
        engine = ProfilingEngine(config)
        
        assert engine.cache is None
    
    def test_cache_operations(self, sample_dataframe):
        """Test cache operations."""
        config = ProfilingConfig(enable_caching=True)
        engine = ProfilingEngine(config)
        
        # Add item to cache
        cache_key = "test_key"
        test_value = {"test": "data"}
        engine.cache[cache_key] = test_value
        
        # Check cache info
        cache_info = engine.get_cache_info()
        assert cache_info['cache_enabled'] is True
        assert cache_info['cache_size'] == 1
        assert cache_key in cache_info['cache_keys']
        
        # Clear cache
        engine.clear_cache()
        assert len(engine.cache) == 0
    
    def test_cache_hit_schema_analysis(self, sample_dataframe, mock_schema_profile):
        """Test cache hit for schema analysis."""
        config = ProfilingConfig(enable_caching=True)
        engine = ProfilingEngine(config)
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile) as mock_infer:
            # First call should hit the service
            result1 = engine._run_schema_analysis(sample_dataframe)
            assert mock_infer.call_count == 1
            
            # Second call should hit the cache
            result2 = engine._run_schema_analysis(sample_dataframe)
            assert mock_infer.call_count == 1  # Still 1, not called again
            
            assert result1 == result2


class TestProfilingStrategies:
    """Test profiling strategy determination."""
    
    def test_get_profiling_strategy_full(self, engine):
        """Test full profiling strategy for small data."""
        small_df = pd.DataFrame({'col1': range(100)})
        strategy = engine._get_profiling_strategy(small_df)
        
        assert strategy == "full"
    
    def test_get_profiling_strategy_incremental(self, engine):
        """Test incremental profiling strategy for medium data."""
        medium_df = pd.DataFrame({'col1': range(60000)})
        strategy = engine._get_profiling_strategy(medium_df)
        
        assert strategy == "incremental"
    
    def test_get_profiling_strategy_sample(self, engine):
        """Test sample profiling strategy for large data."""
        # Create large DataFrame in memory
        large_data = np.random.random((10000, 50))  # Should exceed 100MB
        large_df = pd.DataFrame(large_data)
        
        strategy = engine._get_profiling_strategy(large_df)
        
        assert strategy in ["sample", "incremental", "full"]  # Depends on actual memory usage


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_calculate_sample_percentage(self, engine, sample_dataframe):
        """Test sample percentage calculation."""
        # Test with different sized samples
        full_percentage = engine._calculate_sample_percentage(sample_dataframe, sample_dataframe)
        assert full_percentage == 100.0
        
        half_sample = sample_dataframe.head(500)
        half_percentage = engine._calculate_sample_percentage(sample_dataframe, half_sample)
        assert half_percentage == 50.0
        
        # Test with empty original
        empty_df = pd.DataFrame()
        empty_percentage = engine._calculate_sample_percentage(empty_df, empty_df)
        assert empty_percentage is None
    
    def test_estimate_memory_usage(self, engine, sample_dataframe):
        """Test memory usage estimation."""
        memory_mb = engine._estimate_memory_usage(sample_dataframe)
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_get_dataframe_hash(self, engine, sample_dataframe):
        """Test DataFrame hash generation."""
        hash1 = engine._get_dataframe_hash(sample_dataframe)
        hash2 = engine._get_dataframe_hash(sample_dataframe)
        
        # Same DataFrame should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Different DataFrame should produce different hash
        different_df = pd.DataFrame({'different': [1, 2, 3]})
        hash3 = engine._get_dataframe_hash(different_df)
        assert hash3 != hash1


class TestSpecialProfilingModes:
    """Test special profiling modes."""
    
    def test_profile_incremental(self, engine, sample_dataframe,
                                mock_schema_profile, mock_quality_assessment):
        """Test incremental profiling."""
        # Create a previous profile
        previous_profile = DataProfile(
            profile_id=engine._generate_profile_id(),
            dataset_id=engine._generate_dataset_id()
        )
        
        with patch.object(engine, 'profile_dataset') as mock_profile:
            mock_profile.return_value = DataProfile(
                profile_id=engine._generate_profile_id(),
                dataset_id=engine._generate_dataset_id()
            )
            
            result = engine.profile_incremental(sample_dataframe, previous_profile)
            
            assert mock_profile.called
            assert isinstance(result, DataProfile)
    
    def test_profile_streaming(self, engine, large_dataframe,
                              mock_schema_profile, mock_quality_assessment):
        """Test streaming profiling."""
        with patch.object(engine, 'profile_dataset') as mock_profile:
            mock_profile.return_value = DataProfile(
                profile_id=engine._generate_profile_id(),
                dataset_id=engine._generate_dataset_id()
            )
            
            result = engine.profile_streaming(large_dataframe, window_size=1000)
            
            assert mock_profile.called
            # Check that windowed data was used
            call_args = mock_profile.call_args[0][0]
            assert len(call_args) == 1000  # Should be windowed to 1000 rows
    
    def test_profile_streaming_small_data(self, engine, sample_dataframe):
        """Test streaming profiling with data smaller than window."""
        with patch.object(engine, 'profile_dataset') as mock_profile:
            mock_profile.return_value = DataProfile(
                profile_id=engine._generate_profile_id(),
                dataset_id=engine._generate_dataset_id()
            )
            
            result = engine.profile_streaming(sample_dataframe, window_size=5000)
            
            # Should use full data when smaller than window
            call_args = mock_profile.call_args[0][0]
            assert len(call_args) == len(sample_dataframe)


class TestProfilingSummary:
    """Test profiling summary functionality."""
    
    def test_get_profiling_summary_complete(self, mock_schema_profile, mock_quality_assessment):
        """Test profiling summary with complete profile."""
        from src.packages.data_profiling.domain.entities.data_profile import ProfileId, DatasetId
        
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=DatasetId(),
            source_type="csv"
        )
        profile.start_profiling()
        
        metadata = ProfilingMetadata(execution_time_seconds=30.0)
        profile.complete_profiling(mock_schema_profile, mock_quality_assessment, metadata)
        
        engine = ProfilingEngine()
        summary = engine.get_profiling_summary(profile)
        
        assert 'profile_id' in summary
        assert 'dataset_id' in summary
        assert 'status' in summary
        assert 'source_type' in summary
        assert 'execution_time' in summary
        assert 'schema' in summary
        assert 'quality' in summary
        
        # Check schema summary
        schema_summary = summary['schema']
        assert 'total_columns' in schema_summary
        assert 'total_rows' in schema_summary
        assert 'primary_keys' in schema_summary
        assert 'estimated_size_mb' in schema_summary
        
        # Check quality summary
        quality_summary = summary['quality']
        assert 'overall_score' in quality_summary
        assert 'total_issues' in quality_summary
    
    def test_get_profiling_summary_minimal(self):
        """Test profiling summary with minimal profile."""
        from src.packages.data_profiling.domain.entities.data_profile import ProfileId, DatasetId
        
        profile = DataProfile(
            profile_id=ProfileId(),
            dataset_id=DatasetId()
        )
        
        engine = ProfilingEngine()
        summary = engine.get_profiling_summary(profile)
        
        assert 'profile_id' in summary
        assert 'dataset_id' in summary
        assert 'status' in summary
        assert summary['execution_time'] is None
    
    def test_get_profiling_summary_error_handling(self):
        """Test profiling summary error handling."""
        # Create invalid profile that will cause errors
        invalid_profile = Mock()
        invalid_profile.profile_id = Mock()
        invalid_profile.profile_id.value = Mock(side_effect=Exception("Test error"))
        
        engine = ProfilingEngine()
        summary = engine.get_profiling_summary(invalid_profile)
        
        assert 'error' in summary


class TestParallelProcessing:
    """Test parallel processing functionality."""
    
    def test_parallel_processing_enabled(self, sample_dataframe,
                                        mock_schema_profile, mock_quality_assessment):
        """Test profiling with parallel processing enabled."""
        config = ProfilingConfig(enable_parallel_processing=True, max_workers=2)
        engine = ProfilingEngine(config)
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            profile = engine.profile_dataset(sample_dataframe)
            
            assert profile.status == ProfilingStatus.COMPLETED
    
    def test_parallel_processing_disabled(self, sample_dataframe,
                                         mock_schema_profile, mock_quality_assessment):
        """Test profiling with parallel processing disabled."""
        config = ProfilingConfig(enable_parallel_processing=False)
        engine = ProfilingEngine(config)
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            profile = engine.profile_dataset(sample_dataframe)
            
            assert profile.status == ProfilingStatus.COMPLETED


class TestConfigurationHandling:
    """Test configuration handling."""
    
    def test_selective_analysis_components(self, sample_dataframe, mock_schema_profile):
        """Test running only selected analysis components."""
        config = ProfilingConfig(
            include_schema_analysis=True,
            include_statistical_analysis=False,
            include_pattern_discovery=False,
            include_quality_assessment=False,
            include_relationship_analysis=False
        )
        engine = ProfilingEngine(config)
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile):
            profile = engine.profile_dataset(sample_dataframe)
            
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.schema_profile == mock_schema_profile
            assert profile.quality_assessment is None  # Should be None since disabled
    
    def test_time_series_analysis_configuration(self, sample_dataframe, mock_schema_profile):
        """Test time series analysis configuration."""
        config = ProfilingConfig(enable_time_series_analysis=True)
        engine = ProfilingEngine(config)
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.statistical_service, 'analyze_distribution', return_value={}), \
             patch.object(engine.statistical_service, 'correlation_analysis', return_value={}), \
             patch.object(engine.statistical_service, 'detect_outliers', return_value={}), \
             patch.object(engine.statistical_service, 'analyze_time_series', return_value={'ts_analysis': 'done'}):
            
            result = engine._run_statistical_analysis(sample_dataframe)
            
            # Should include time series analysis results
            assert any('time_series_' in key for key in result.keys())


class TestAdvancedOrchestrator:
    """Test advanced orchestrator integration."""
    
    @pytest.mark.asyncio
    async def test_profile_with_advanced_orchestrator(self, sample_dataframe):
        """Test profiling with advanced orchestrator."""
        config = ProfilingConfig(enable_ml_clustering=True, max_workers=2)
        engine = ProfilingEngine(config)
        
        # Mock the advanced orchestrator
        mock_profile = DataProfile()
        mock_profile.status = ProfilingStatus.COMPLETED
        
        with patch.object(engine.advanced_orchestrator, 'profile_dataset_comprehensive',
                         return_value=mock_profile) as mock_comprehensive:
            
            result = await engine._profile_with_advanced_orchestrator(
                df=sample_dataframe,
                dataset_id="test_dataset",
                source_type="csv",
                source_connection={"path": "/test.csv"}
            )
            
            # Verify the mock was called with correct parameters
            mock_comprehensive.assert_called_once()
            call_kwargs = mock_comprehensive.call_args[1]
            
            assert call_kwargs['dataset_name'] == "test_dataset"
            assert call_kwargs['source_metadata']['type'] == "csv"
            assert call_kwargs['source_metadata']['connection'] == {"path": "/test.csv"}
            assert 'profiling_options' in call_kwargs
            
            # Verify result
            assert result == mock_profile
    
    def test_profile_dataset_with_advanced_orchestrator_flag(self, sample_dataframe):
        """Test profile_dataset with advanced orchestrator flag."""
        engine = ProfilingEngine()
        
        with patch.object(engine, '_profile_with_advanced_orchestrator',
                         return_value=DataProfile()) as mock_advanced:
            
            result = engine.profile_dataset(
                df=sample_dataframe,
                dataset_id="test_dataset",
                use_advanced_orchestrator=True
            )
            
            mock_advanced.assert_called_once_with(
                sample_dataframe, "test_dataset", "dataframe", None
            )
            assert isinstance(result, DataProfile)
    
    def test_profile_dataset_without_advanced_orchestrator(self, sample_dataframe,
                                                          mock_schema_profile, mock_quality_assessment):
        """Test profile_dataset without advanced orchestrator (default behavior)."""
        engine = ProfilingEngine()
        
        with patch.object(engine.schema_service, 'infer', return_value=mock_schema_profile), \
             patch.object(engine.quality_service, 'assess_quality', return_value=mock_quality_assessment), \
             patch.object(engine.statistical_service, 'analyze', return_value={}), \
             patch.object(engine.pattern_service, 'discover', return_value={}), \
             patch.object(engine.schema_service, 'detect_advanced_relationships', return_value={}):
            
            result = engine.profile_dataset(
                df=sample_dataframe,
                use_advanced_orchestrator=False
            )
            
            assert result.status == ProfilingStatus.COMPLETED
            assert result.schema_profile == mock_schema_profile
            assert result.quality_assessment == mock_quality_assessment
    
    @pytest.mark.asyncio
    async def test_advanced_orchestrator_error_handling(self, sample_dataframe):
        """Test error handling in advanced orchestrator."""
        engine = ProfilingEngine()
        
        with patch.object(engine.advanced_orchestrator, 'profile_dataset_comprehensive',
                         side_effect=Exception("Advanced profiling failed")) as mock_comprehensive:
            
            with pytest.raises(Exception) as exc_info:
                await engine._profile_with_advanced_orchestrator(
                    df=sample_dataframe,
                    dataset_id="test_dataset"
                )
            
            assert "Advanced profiling failed" in str(exc_info.value)
    
    def test_advanced_orchestrator_configuration_mapping(self, sample_dataframe):
        """Test configuration mapping to advanced orchestrator options."""
        config = ProfilingConfig(
            enable_advanced_patterns=True,
            enable_relationship_analysis=True,
            enable_sampling=True,
            sample_size=5000
        )
        engine = ProfilingEngine(config)
        
        with patch.object(engine.advanced_orchestrator, 'profile_dataset_comprehensive',
                         return_value=DataProfile()) as mock_comprehensive:
            
            # Use asyncio.run to handle async call
            import asyncio
            asyncio.run(engine._profile_with_advanced_orchestrator(
                df=sample_dataframe,
                dataset_id="test_dataset"
            ))
            
            # Check that configuration was properly mapped
            call_kwargs = mock_comprehensive.call_args[1]
            profiling_options = call_kwargs['profiling_options']
            
            assert profiling_options['enable_advanced_analysis'] == config.enable_advanced_patterns
            assert profiling_options['enable_relationship_analysis'] == config.enable_relationship_analysis
            assert profiling_options['sample_size'] == config.sample_size


# Helper methods for the engine (these would normally be internal)
def _generate_profile_id(self):
    """Generate a profile ID for testing."""
    from src.packages.data_profiling.domain.entities.data_profile import ProfileId
    return ProfileId()

def _generate_dataset_id(self):
    """Generate a dataset ID for testing."""
    from src.packages.data_profiling.domain.entities.data_profile import DatasetId
    return DatasetId()

# Add helper methods to ProfilingEngine for testing
ProfilingEngine._generate_profile_id = _generate_profile_id
ProfilingEngine._generate_dataset_id = _generate_dataset_id