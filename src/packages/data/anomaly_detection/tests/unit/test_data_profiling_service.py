"""Comprehensive test suite for DataProfilingService."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from anomaly_detection.domain.services.data_profiling_service import DataProfilingService


class TestDataProfilingService:
    """Test suite for DataProfilingService."""
    
    @pytest.fixture
    def profiling_service(self):
        """Create DataProfilingService instance."""
        return DataProfilingService()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data with various characteristics."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'numeric_normal': np.random.normal(0, 1, 1000),
            'numeric_integer': np.random.randint(1, 100, 1000),
            'numeric_with_missing': np.concatenate([
                np.random.normal(5, 2, 900),
                [np.nan] * 100
            ]),
            'categorical': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'boolean': np.random.choice([True, False], 1000),
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'text': [f"text_{i}" for i in range(1000)],
            'constant': [42] * 1000,
            'unique_id': range(1000)
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield Path(f.name)
            Path(f.name).unlink()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_profile_sections(self, profiling_service):
        """Test initialization with profile sections."""
        expected_sections = [
            'dataset_info', 'column_info', 'data_quality',
            'statistical_summary', 'correlations', 'distributions', 'patterns'
        ]
        
        for section in expected_sections:
            assert section in profiling_service.profile_sections
            assert callable(profiling_service.profile_sections[section])

    @pytest.mark.asyncio
    async def test_profile_file_complete(self, profiling_service, temp_csv_file):
        """Test complete file profiling."""
        profile = await profiling_service.profile_file(temp_csv_file)
        
        # Check main sections exist
        assert 'dataset_info' in profile
        assert 'column_info' in profile
        assert 'data_quality' in profile
        assert 'statistical_summary' in profile
        assert 'correlations' in profile
        assert 'distributions' in profile
        assert 'patterns' in profile
        
        # Check dataset info
        assert profile['dataset_info']['row_count'] == 1000
        assert profile['dataset_info']['column_count'] == 9

    @pytest.mark.asyncio
    async def test_profile_file_selective_sections(self, profiling_service, temp_csv_file):
        """Test profiling with selective sections."""
        sections = ['dataset_info', 'data_quality']
        
        profile = await profiling_service.profile_file(
            temp_csv_file,
            sections=sections
        )
        
        # Should only include requested sections
        assert 'dataset_info' in profile
        assert 'data_quality' in profile
        assert 'correlations' not in profile
        assert 'distributions' not in profile

    @pytest.mark.asyncio
    async def test_profile_file_with_sampling(self, profiling_service, temp_csv_file):
        """Test profiling with data sampling."""
        profile = await profiling_service.profile_file(
            temp_csv_file,
            sample_size=500
        )
        
        # Should process sampled data
        assert profile['dataset_info']['row_count'] == 500
        assert profile['dataset_info']['column_count'] == 9

    @pytest.mark.asyncio
    async def test_profile_file_nonexistent(self, profiling_service):
        """Test profiling nonexistent file."""
        nonexistent_file = Path("nonexistent.csv")
        
        profile = await profiling_service.profile_file(nonexistent_file)
        
        assert profile['status'] == 'error'
        assert 'error' in profile
        assert 'not found' in profile['error'].lower()

    def test_profile_dataset_info(self, profiling_service, sample_data):
        """Test dataset information profiling."""
        info = profiling_service._profile_dataset_info(sample_data)
        
        assert info['row_count'] == 1000
        assert info['column_count'] == 9
        assert info['memory_usage_mb'] > 0
        assert 'creation_time' in info
        assert 'data_types' in info

    def test_profile_columns(self, profiling_service, sample_data):
        """Test column-level profiling."""
        column_info = profiling_service._profile_columns(sample_data)
        
        # Should have info for each column
        assert len(column_info) == 9
        
        # Check numeric column
        numeric_col = column_info['numeric_normal']
        assert numeric_col['data_type'] == 'float64'
        assert numeric_col['non_null_count'] == 1000
        assert numeric_col['null_count'] == 0
        assert 'mean' in numeric_col
        assert 'std' in numeric_col
        
        # Check categorical column
        cat_col = column_info['categorical']
        assert cat_col['data_type'] == 'object'
        assert cat_col['unique_count'] == 4
        assert 'value_counts' in cat_col

    def test_profile_data_quality(self, profiling_service, sample_data):
        """Test data quality profiling."""
        quality = profiling_service._profile_data_quality(sample_data)
        
        assert 'missing_values' in quality
        assert 'duplicate_rows' in quality
        assert 'data_completeness' in quality
        assert 'consistency_issues' in quality
        
        # Check missing values detection
        assert quality['missing_values']['numeric_with_missing'] == 100
        assert quality['missing_values']['numeric_normal'] == 0

    def test_profile_statistics(self, profiling_service, sample_data):
        """Test statistical summary profiling."""
        stats = profiling_service._profile_statistics(sample_data)
        
        # Should have stats for numeric columns
        assert 'numeric_normal' in stats
        assert 'numeric_integer' in stats
        assert 'numeric_with_missing' in stats
        
        # Check statistical measures
        numeric_stats = stats['numeric_normal']
        assert 'mean' in numeric_stats
        assert 'median' in numeric_stats
        assert 'std' in numeric_stats
        assert 'min' in numeric_stats
        assert 'max' in numeric_stats
        assert 'percentiles' in numeric_stats

    def test_profile_correlations(self, profiling_service, sample_data):
        """Test correlation analysis."""
        correlations = profiling_service._profile_correlations(sample_data)
        
        assert 'correlation_matrix' in correlations
        assert 'high_correlations' in correlations
        
        # Correlation matrix should be square
        corr_matrix = correlations['correlation_matrix']
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        assert len(corr_matrix) == len(numeric_cols)

    def test_profile_distributions(self, profiling_service, sample_data):
        """Test distribution analysis."""
        distributions = profiling_service._profile_distributions(sample_data)
        
        # Should analyze numeric columns
        assert 'numeric_normal' in distributions
        assert 'numeric_integer' in distributions
        
        # Check distribution properties
        normal_dist = distributions['numeric_normal']
        assert 'histogram' in normal_dist
        assert 'distribution_type' in normal_dist
        assert 'normality_test' in normal_dist

    def test_profile_patterns(self, profiling_service, sample_data):
        """Test pattern analysis."""
        patterns = profiling_service._profile_patterns(sample_data)
        
        assert 'constant_columns' in patterns
        assert 'unique_columns' in patterns
        assert 'text_patterns' in patterns
        
        # Should detect constant column
        assert 'constant' in patterns['constant_columns']
        
        # Should detect unique column
        assert 'unique_id' in patterns['unique_columns']

    def test_detect_outliers(self, profiling_service):
        """Test outlier detection."""
        # Create data with clear outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, -100, 2])
        
        outliers = profiling_service._detect_outliers(data_with_outliers)
        
        assert len(outliers) > 0
        assert 100 in outliers or -100 in outliers

    def test_detect_outliers_no_outliers(self, profiling_service):
        """Test outlier detection with clean data."""
        clean_data = pd.Series(np.random.normal(0, 1, 1000))
        
        outliers = profiling_service._detect_outliers(clean_data)
        
        # Should have very few outliers in normal distribution
        assert len(outliers) < 50  # Less than 5% outliers

    def test_analyze_text_patterns(self, profiling_service):
        """Test text pattern analysis."""
        text_data = pd.Series([
            'email@domain.com', 'test@example.org', 'user@site.net',
            '123-45-6789', '987-65-4321',
            'ABC123', 'DEF456', 'GHI789'
        ])
        
        patterns = profiling_service._analyze_text_patterns(text_data)
        
        assert 'email_pattern' in patterns
        assert 'phone_pattern' in patterns
        assert 'alphanumeric_pattern' in patterns
        
        assert patterns['email_pattern'] == 3
        assert patterns['phone_pattern'] == 2

    def test_analyze_temporal_patterns(self, profiling_service):
        """Test temporal pattern analysis."""
        date_data = pd.Series(pd.date_range('2023-01-01', periods=365, freq='D'))
        
        patterns = profiling_service._analyze_temporal_patterns(date_data)
        
        assert 'date_range' in patterns
        assert 'frequency' in patterns
        assert 'seasonality' in patterns
        assert 'gaps' in patterns

    def test_calculate_data_quality_score(self, profiling_service, sample_data):
        """Test data quality score calculation."""
        score = profiling_service._calculate_data_quality_score(sample_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))

    def test_generate_recommendations(self, profiling_service, sample_data):
        """Test recommendation generation."""
        profile = {
            'data_quality': profiling_service._profile_data_quality(sample_data),
            'patterns': profiling_service._profile_patterns(sample_data),
            'statistical_summary': profiling_service._profile_statistics(sample_data)
        }
        
        recommendations = profiling_service._generate_recommendations(profile)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend handling missing values
        rec_text = ' '.join(recommendations).lower()
        assert 'missing' in rec_text

    @pytest.mark.asyncio
    async def test_profile_multiple_files(self, profiling_service, sample_data, temp_dir):
        """Test profiling multiple files."""
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = temp_dir / f"data_{i}.csv"
            sample_data.to_csv(file_path, index=False)
            files.append(file_path)
        
        profiles = await profiling_service.profile_multiple_files(files)
        
        assert len(profiles) == 3
        for profile in profiles:
            assert 'dataset_info' in profile
            assert 'data_quality' in profile

    @pytest.mark.asyncio
    async def test_profile_directory(self, profiling_service, sample_data, temp_dir):
        """Test profiling entire directory."""
        # Create test files in directory
        for i in range(3):
            file_path = temp_dir / f"data_{i}.csv"
            sample_data.to_csv(file_path, index=False)
        
        profiles = await profiling_service.profile_directory(
            temp_dir,
            pattern="*.csv"
        )
        
        assert len(profiles) >= 3
        for profile in profiles:
            assert 'file_path' in profile
            assert 'dataset_info' in profile

    def test_compare_profiles(self, profiling_service, sample_data):
        """Test profile comparison."""
        # Create two similar datasets
        data1 = sample_data.copy()
        data2 = sample_data.copy()
        data2['numeric_normal'] = data2['numeric_normal'] * 2  # Scale one column
        
        profile1 = {
            'dataset_info': profiling_service._profile_dataset_info(data1),
            'statistical_summary': profiling_service._profile_statistics(data1)
        }
        
        profile2 = {
            'dataset_info': profiling_service._profile_dataset_info(data2),
            'statistical_summary': profiling_service._profile_statistics(data2)
        }
        
        comparison = profiling_service.compare_profiles(profile1, profile2)
        
        assert 'differences' in comparison
        assert 'similarities' in comparison
        assert 'recommendations' in comparison

    def test_generate_html_report(self, profiling_service, sample_data, temp_dir):
        """Test HTML report generation."""
        profile = {
            'dataset_info': profiling_service._profile_dataset_info(sample_data),
            'column_info': profiling_service._profile_columns(sample_data),
            'data_quality': profiling_service._profile_data_quality(sample_data)
        }
        
        report_file = profiling_service.generate_html_report(profile, temp_dir)
        
        assert report_file.exists()
        assert report_file.suffix == '.html'
        
        # Verify content
        content = report_file.read_text()
        assert '<html>' in content
        assert 'Data Profile Report' in content

    def test_export_profile_json(self, profiling_service, sample_data, temp_dir):
        """Test JSON profile export."""
        profile = {
            'dataset_info': profiling_service._profile_dataset_info(sample_data),
            'data_quality': profiling_service._profile_data_quality(sample_data)
        }
        
        json_file = profiling_service.export_profile(profile, temp_dir, format='json')
        
        assert json_file.exists()
        assert json_file.suffix == '.json'
        
        # Verify content
        with open(json_file) as f:
            loaded_profile = json.load(f)
            assert 'dataset_info' in loaded_profile

    def test_detect_data_drift(self, profiling_service, sample_data):
        """Test data drift detection."""
        # Create reference and current profiles
        reference_data = sample_data.copy()
        current_data = sample_data.copy()
        
        # Introduce drift
        current_data['numeric_normal'] = current_data['numeric_normal'] + 2
        
        ref_profile = profiling_service._profile_statistics(reference_data)
        curr_profile = profiling_service._profile_statistics(current_data)
        
        drift_report = profiling_service.detect_data_drift(ref_profile, curr_profile)
        
        assert 'drift_detected' in drift_report
        assert 'drifted_columns' in drift_report
        assert 'drift_scores' in drift_report
        
        # Should detect drift in numeric_normal column
        assert 'numeric_normal' in drift_report['drifted_columns']

    def test_memory_usage_calculation(self, profiling_service, sample_data):
        """Test memory usage calculation."""
        memory_info = profiling_service._calculate_memory_usage(sample_data)
        
        assert 'total_memory_mb' in memory_info
        assert 'memory_per_column' in memory_info
        assert 'memory_breakdown' in memory_info
        
        assert memory_info['total_memory_mb'] > 0

    @pytest.mark.asyncio
    async def test_streaming_profile_update(self, profiling_service, sample_data):
        """Test streaming profile updates."""
        # Initialize with first batch
        initial_profile = profiling_service._profile_dataset_info(sample_data[:500])
        
        # Update with second batch
        updated_profile = await profiling_service.update_streaming_profile(
            initial_profile, 
            sample_data[500:]
        )
        
        assert updated_profile['row_count'] == 1000
        assert 'streaming_updates' in updated_profile

    def test_categorical_analysis(self, profiling_service):
        """Test categorical data analysis."""
        categorical_data = pd.Series(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'A'])
        
        analysis = profiling_service._analyze_categorical(categorical_data)
        
        assert 'unique_count' in analysis
        assert 'value_counts' in analysis
        assert 'entropy' in analysis
        assert analysis['unique_count'] == 3

    def test_anomaly_detection_in_profiling(self, profiling_service):
        """Test anomaly detection during profiling."""
        # Create data with anomalies
        normal_data = np.random.normal(0, 1, 1000)
        anomalous_data = np.concatenate([normal_data, [10, -10, 15]])
        data = pd.Series(anomalous_data)
        
        anomalies = profiling_service._detect_anomalies_in_column(data)
        
        assert len(anomalies) > 0
        assert any(abs(val) > 5 for val in anomalies)

    def test_profile_caching(self, profiling_service, temp_dir):
        """Test profile result caching."""
        cache_dir = temp_dir / "cache"
        profiling_service.enable_caching(cache_dir)
        
        # Create test data
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        # First profiling should create cache
        profile1 = profiling_service._profile_dataset_info(test_data)
        cache_file = cache_dir / "profile_cache.json"
        
        # Second profiling should use cache (if implemented)
        profile2 = profiling_service._profile_dataset_info(test_data)
        
        assert profile1 == profile2

    @pytest.mark.asyncio
    async def test_performance_large_dataset(self, profiling_service, temp_dir):
        """Test profiling performance on large dataset."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': np.random.rand(50000),
            'col2': np.random.randint(0, 1000, 50000),
            'col3': np.random.choice(['A', 'B', 'C', 'D'], 50000)
        })
        
        large_file = temp_dir / "large_data.csv"
        large_data.to_csv(large_file, index=False)
        
        import time
        start_time = time.time()
        
        profile = await profiling_service.profile_file(
            large_file,
            sample_size=10000  # Use sampling for performance
        )
        
        processing_time = time.time() - start_time
        
        assert profile['dataset_info']['row_count'] == 10000
        assert processing_time < 60  # Should complete within reasonable time

    def teardown_method(self):
        """Cleanup after each test."""
        pass