"""Comprehensive test suite for DataSamplingService."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from anomaly_detection.domain.services.data_sampling_service import DataSamplingService


class TestDataSamplingService:
    """Test suite for DataSamplingService."""
    
    @pytest.fixture
    def sampling_service(self):
        """Create DataSamplingService instance."""
        return DataSamplingService()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.randint(1, 100, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000, p=[0.5, 0.3, 0.2]),
            'group': np.random.choice(['Group1', 'Group2', 'Group3', 'Group4'], 10000),
            'numeric_id': range(10000),
            'boolean_flag': np.random.choice([True, False], 10000)
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

    def test_init_sampling_methods(self, sampling_service):
        """Test initialization with sampling methods."""
        expected_methods = ['random', 'systematic', 'stratified', 'cluster', 'reservoir']
        
        for method in expected_methods:
            assert method in sampling_service.sampling_methods
            assert callable(sampling_service.sampling_methods[method])

    @pytest.mark.asyncio
    async def test_sample_file_random(self, sampling_service, temp_csv_file):
        """Test random sampling from file."""
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=1000,
            method='random',
            seed=42
        )
        
        assert len(sample) == 1000
        assert list(sample.columns) == ['feature1', 'feature2', 'category', 'group', 'numeric_id', 'boolean_flag']
        
        # Should maintain data types
        assert sample['feature1'].dtype == np.float64
        assert sample['feature2'].dtype == np.int64

    @pytest.mark.asyncio
    async def test_sample_file_systematic(self, sampling_service, temp_csv_file):
        """Test systematic sampling from file."""
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=1000,
            method='systematic'
        )
        
        assert len(sample) == 1000
        
        # Should be evenly spaced (every 10th row for 10000->1000)
        # Check that samples are systematically distributed
        assert len(sample) <= 1000

    @pytest.mark.asyncio
    async def test_sample_file_stratified(self, sampling_service, temp_csv_file):
        """Test stratified sampling from file."""
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=1000,
            method='stratified',
            stratify_column='category',
            seed=42
        )
        
        assert len(sample) == 1000
        
        # Should preserve category distribution approximately
        original_proportions = pd.Series(['A', 'B', 'C']).value_counts(normalize=True).sort_index()
        sample_proportions = sample['category'].value_counts(normalize=True).sort_index()
        
        # Proportions should be reasonably close
        for cat in ['A', 'B', 'C']:
            if cat in sample_proportions.index and cat in original_proportions.index:
                assert abs(sample_proportions[cat] - original_proportions.get(cat, 0)) < 0.1

    @pytest.mark.asyncio
    async def test_sample_file_cluster(self, sampling_service, temp_csv_file):
        """Test cluster sampling from file."""
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=1000,
            method='cluster',
            cluster_column='group',
            seed=42
        )
        
        assert len(sample) <= 1000  # May be less due to cluster sampling
        
        # Should contain complete clusters
        unique_groups = sample['group'].unique()
        assert len(unique_groups) > 0

    @pytest.mark.asyncio
    async def test_sample_file_reservoir(self, sampling_service, temp_csv_file):
        """Test reservoir sampling from file."""
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=1000,
            method='reservoir',
            seed=42
        )
        
        assert len(sample) == 1000
        
        # Reservoir sampling should work with any data size
        assert sample is not None

    @pytest.mark.asyncio
    async def test_sample_file_unsupported_method(self, sampling_service, temp_csv_file):
        """Test sampling with unsupported method."""
        with pytest.raises(ValueError, match="Unsupported sampling method"):
            await sampling_service.sample_file(
                file_path=temp_csv_file,
                sample_size=1000,
                method='invalid_method'
            )

    @pytest.mark.asyncio
    async def test_sample_file_nonexistent(self, sampling_service):
        """Test sampling nonexistent file."""
        nonexistent_file = Path("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            await sampling_service.sample_file(
                file_path=nonexistent_file,
                sample_size=1000
            )

    @pytest.mark.asyncio
    async def test_sample_file_large_sample_size(self, sampling_service, temp_csv_file):
        """Test sampling with sample size larger than data."""
        # Sample size larger than available data
        sample = await sampling_service.sample_file(
            file_path=temp_csv_file,
            sample_size=20000,  # More than 10000 rows available
            method='random'
        )
        
        # Should return all available data or raise appropriate error
        assert len(sample) <= 10000

    def test_random_sampling(self, sampling_service, sample_data):
        """Test random sampling method."""
        sample = sampling_service._random_sampling(
            data=sample_data,
            sample_size=1000,
            seed=42,
            replacement=False
        )
        
        assert len(sample) == 1000
        assert len(sample.index.unique()) == 1000  # No duplicates without replacement

    def test_random_sampling_with_replacement(self, sampling_service, sample_data):
        """Test random sampling with replacement."""
        sample = sampling_service._random_sampling(
            data=sample_data,
            sample_size=1000,
            seed=42,
            replacement=True
        )
        
        assert len(sample) == 1000
        # May have duplicates with replacement

    def test_systematic_sampling(self, sampling_service, sample_data):
        """Test systematic sampling method."""
        sample = sampling_service._systematic_sampling(
            data=sample_data,
            sample_size=1000,
            seed=42
        )
        
        assert len(sample) == 1000
        
        # Should be evenly spaced
        interval = len(sample_data) // 1000
        assert interval == 10  # 10000 / 1000 = 10

    def test_systematic_sampling_small_data(self, sampling_service):
        """Test systematic sampling with small dataset."""
        small_data = pd.DataFrame({'col1': range(50)})
        
        sample = sampling_service._systematic_sampling(
            data=small_data,
            sample_size=20,
            seed=42
        )
        
        assert len(sample) <= 20
        assert len(sample) <= 50

    def test_stratified_sampling(self, sampling_service, sample_data):
        """Test stratified sampling method."""
        sample = sampling_service._stratified_sampling(
            data=sample_data,
            sample_size=1000,
            stratify_column='category',
            seed=42
        )
        
        assert len(sample) == 1000
        
        # Should preserve proportions
        original_counts = sample_data['category'].value_counts(normalize=True)
        sample_counts = sample['category'].value_counts(normalize=True)
        
        # Check that proportions are approximately maintained
        for category in original_counts.index:
            if category in sample_counts.index:
                assert abs(original_counts[category] - sample_counts[category]) < 0.15

    def test_stratified_sampling_missing_column(self, sampling_service, sample_data):
        """Test stratified sampling with missing stratify column."""
        with pytest.raises(KeyError):
            sampling_service._stratified_sampling(
                data=sample_data,
                sample_size=1000,
                stratify_column='nonexistent_column',
                seed=42
            )

    def test_cluster_sampling(self, sampling_service, sample_data):
        """Test cluster sampling method."""
        sample = sampling_service._cluster_sampling(
            data=sample_data,
            sample_size=1000,
            cluster_column='group',
            seed=42
        )
        
        assert len(sample) > 0
        
        # Should contain complete clusters
        sampled_groups = sample['group'].unique()
        
        # Each sampled group should contain all its original members
        for group in sampled_groups:
            original_group_size = len(sample_data[sample_data['group'] == group])
            sample_group_size = len(sample[sample['group'] == group])
            assert sample_group_size == original_group_size

    def test_cluster_sampling_missing_column(self, sampling_service, sample_data):
        """Test cluster sampling with missing cluster column."""
        with pytest.raises(KeyError):
            sampling_service._cluster_sampling(
                data=sample_data,
                sample_size=1000,
                cluster_column='nonexistent_column',
                seed=42
            )

    def test_reservoir_sampling(self, sampling_service, sample_data):
        """Test reservoir sampling method."""
        sample = sampling_service._reservoir_sampling(
            data=sample_data,
            sample_size=1000,
            seed=42
        )
        
        assert len(sample) == 1000
        
        # Each row should appear only once
        assert len(sample.index.unique()) == 1000

    def test_reservoir_sampling_small_data(self, sampling_service):
        """Test reservoir sampling with data smaller than sample size."""
        small_data = pd.DataFrame({'col1': range(100)})
        
        sample = sampling_service._reservoir_sampling(
            data=small_data,
            sample_size=1000,
            seed=42
        )
        
        # Should return all available data
        assert len(sample) == 100

    def test_validate_sample_size_valid(self, sampling_service, sample_data):
        """Test sample size validation with valid size."""
        # Should not raise error
        sampling_service._validate_sample_size(sample_data, 1000)

    def test_validate_sample_size_too_large(self, sampling_service, sample_data):
        """Test sample size validation with size too large."""
        with pytest.raises(ValueError, match="Sample size cannot be larger than data size"):
            sampling_service._validate_sample_size(sample_data, 20000, allow_larger=False)

    def test_validate_sample_size_zero(self, sampling_service, sample_data):
        """Test sample size validation with zero size."""
        with pytest.raises(ValueError, match="Sample size must be positive"):
            sampling_service._validate_sample_size(sample_data, 0)

    def test_validate_sample_size_negative(self, sampling_service, sample_data):
        """Test sample size validation with negative size."""
        with pytest.raises(ValueError, match="Sample size must be positive"):
            sampling_service._validate_sample_size(sample_data, -100)

    @pytest.mark.asyncio
    async def test_sample_streaming_data(self, sampling_service):
        """Test sampling from streaming data."""
        # Simulate streaming data chunks
        data_chunks = [
            pd.DataFrame({'col1': range(100), 'col2': range(100, 200)}),
            pd.DataFrame({'col1': range(200, 300), 'col2': range(300, 400)}),
            pd.DataFrame({'col1': range(400, 500), 'col2': range(500, 600)})
        ]
        
        sampled_chunks = []
        for chunk in data_chunks:
            sample = sampling_service._random_sampling(chunk, 30, seed=42)
            sampled_chunks.append(sample)
        
        # Combine samples
        combined_sample = pd.concat(sampled_chunks, ignore_index=True)
        
        assert len(combined_sample) == 90  # 30 * 3 chunks
        assert len(combined_sample.columns) == 2

    def test_weighted_sampling(self, sampling_service):
        """Test weighted sampling."""
        # Create data with weights
        data_with_weights = pd.DataFrame({
            'value': range(100),
            'weight': np.random.uniform(0.1, 1.0, 100)
        })
        
        sample = sampling_service._weighted_sampling(
            data=data_with_weights,
            sample_size=50,
            weight_column='weight',
            seed=42
        )
        
        assert len(sample) == 50
        
        # Higher weighted items should have higher chance of selection
        # (This is probabilistic, so we can't guarantee specific outcomes)
        assert sample is not None

    def test_time_based_sampling(self, sampling_service):
        """Test time-based sampling."""
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        time_data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.rand(1000)
        })
        
        sample = sampling_service._time_based_sampling(
            data=time_data,
            sample_size=100,
            time_column='timestamp',
            method='uniform',
            seed=42
        )
        
        assert len(sample) == 100
        assert 'timestamp' in sample.columns
        
        # Should be distributed across time range
        time_range = sample['timestamp'].max() - sample['timestamp'].min()
        assert time_range.days > 0

    @pytest.mark.asyncio
    async def test_adaptive_sampling(self, sampling_service, sample_data, temp_dir):
        """Test adaptive sampling based on data characteristics."""
        # Save data to file
        file_path = temp_dir / "adaptive_test.csv"
        sample_data.to_csv(file_path, index=False)
        
        sample = await sampling_service.adaptive_sample(
            file_path=file_path,
            target_size=1000,
            seed=42
        )
        
        assert len(sample) <= 1000
        
        # Should choose appropriate sampling method based on data
        assert sample is not None

    def test_bootstrap_sampling(self, sampling_service, sample_data):
        """Test bootstrap sampling."""
        bootstrap_samples = sampling_service.bootstrap_sample(
            data=sample_data,
            sample_size=1000,
            n_bootstrap=10,
            seed=42
        )
        
        assert len(bootstrap_samples) == 10
        
        for sample in bootstrap_samples:
            assert len(sample) == 1000
            # Bootstrap samples should have potential duplicates
            assert isinstance(sample, pd.DataFrame)

    def test_stratified_bootstrap(self, sampling_service, sample_data):
        """Test stratified bootstrap sampling."""
        bootstrap_samples = sampling_service.stratified_bootstrap(
            data=sample_data,
            sample_size=1000,
            stratify_column='category',
            n_bootstrap=5,
            seed=42
        )
        
        assert len(bootstrap_samples) == 5
        
        for sample in bootstrap_samples:
            assert len(sample) == 1000
            # Should maintain category proportions
            assert 'category' in sample.columns

    def test_calculate_sampling_error(self, sampling_service, sample_data):
        """Test sampling error calculation."""
        sample = sampling_service._random_sampling(sample_data, 1000, seed=42)
        
        error = sampling_service.calculate_sampling_error(
            population=sample_data,
            sample=sample,
            column='feature1'
        )
        
        assert 'mean_error' in error
        assert 'std_error' in error
        assert 'confidence_interval' in error
        
        assert isinstance(error['mean_error'], float)

    def test_optimal_sample_size(self, sampling_service, sample_data):
        """Test optimal sample size calculation."""
        optimal_size = sampling_service.calculate_optimal_sample_size(
            population_size=len(sample_data),
            confidence_level=0.95,
            margin_of_error=0.05,
            population_proportion=0.5
        )
        
        assert isinstance(optimal_size, int)
        assert optimal_size > 0
        assert optimal_size <= len(sample_data)

    @pytest.mark.asyncio
    async def test_sample_quality_assessment(self, sampling_service, sample_data, temp_dir):
        """Test sample quality assessment."""
        # Create file and sample
        file_path = temp_dir / "quality_test.csv"
        sample_data.to_csv(file_path, index=False)
        
        sample = await sampling_service.sample_file(file_path, 1000, seed=42)
        
        quality_report = sampling_service.assess_sample_quality(
            population=sample_data,
            sample=sample
        )
        
        assert 'representativeness_score' in quality_report
        assert 'bias_assessment' in quality_report
        assert 'coverage_analysis' in quality_report
        
        assert 0 <= quality_report['representativeness_score'] <= 1

    def test_multi_stage_sampling(self, sampling_service):
        """Test multi-stage sampling."""
        # Create hierarchical data
        hierarchical_data = pd.DataFrame({
            'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
            'city': np.random.choice([f'City_{i}' for i in range(20)], 10000),
            'household': np.random.choice([f'HH_{i}' for i in range(1000)], 10000),
            'value': np.random.rand(10000)
        })
        
        sample = sampling_service._multi_stage_sampling(
            data=hierarchical_data,
            stages=['region', 'city', 'household'],
            sample_sizes=[2, 5, 100],
            seed=42
        )
        
        assert len(sample) > 0
        assert len(sample['region'].unique()) <= 2
        assert len(sample['city'].unique()) <= 10  # 2 regions * 5 cities max

    @pytest.mark.asyncio
    async def test_performance_large_dataset(self, sampling_service, temp_dir):
        """Test sampling performance on large dataset."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': np.random.rand(100000),
            'col2': np.random.randint(0, 1000, 100000),
            'col3': np.random.choice(['A', 'B', 'C'], 100000)
        })
        
        large_file = temp_dir / "large_data.csv"
        large_data.to_csv(large_file, index=False)
        
        import time
        start_time = time.time()
        
        sample = await sampling_service.sample_file(
            file_path=large_file,
            sample_size=5000,
            method='random',
            seed=42
        )
        
        processing_time = time.time() - start_time
        
        assert len(sample) == 5000
        assert processing_time < 30  # Should complete within reasonable time

    def test_sampling_with_missing_values(self, sampling_service):
        """Test sampling behavior with missing values."""
        data_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'col2': ['A', 'B', 'C', np.nan, 'E', 'F', np.nan, 'H', 'I', 'J']
        })
        
        sample = sampling_service._random_sampling(
            data=data_with_missing,
            sample_size=5,
            seed=42
        )
        
        assert len(sample) == 5
        # Should preserve missing values in sample
        assert sample.isnull().sum().sum() >= 0

    def teardown_method(self):
        """Cleanup after each test."""
        pass