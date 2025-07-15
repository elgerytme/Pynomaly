import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
from ..application.use_cases.profile_dataset import ProfileDatasetUseCase
from ..application.services.performance_optimizer import PerformanceOptimizer, SamplingStrategy
from ..infrastructure.adapters.file_adapter import get_file_adapter
from ..domain.entities.data_profile import ProfilingStatus


class TestProfileDatasetUseCase:
    
    def setup_method(self):
        self.use_case = ProfileDatasetUseCase()
        
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 22],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 45000.0],
            'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing'],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com', 'eve@example.com']
        })
    
    def test_execute_full_profiling(self):
        """Test full dataset profiling."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Execute profiling
            profile = self.use_case.execute(temp_path, "full")
            
            # Verify profile structure
            assert profile is not None
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.schema_profile is not None
            assert profile.quality_assessment is not None
            assert profile.profiling_metadata is not None
            
            # Verify schema profile
            schema = profile.schema_profile
            assert schema.total_rows == 5
            assert schema.total_columns == 6
            assert len(schema.columns) == 6
            
            # Verify column profiles
            column_names = [col.column_name for col in schema.columns]
            assert 'id' in column_names
            assert 'name' in column_names
            assert 'age' in column_names
            assert 'salary' in column_names
            assert 'department' in column_names
            assert 'email' in column_names
            
            # Verify patterns were discovered
            email_column = next((col for col in schema.columns if col.column_name == 'email'), None)
            assert email_column is not None
            assert len(email_column.patterns) > 0  # Should detect email pattern
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_execute_sample_profiling(self):
        """Test sample-based profiling."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Execute profiling with sampling
            profile = self.use_case.execute_sample(temp_path, sample_size=3)
            
            # Verify profile structure
            assert profile is not None
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.profiling_metadata.sample_size == 3
            assert profile.profiling_metadata.profiling_strategy == "sample"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_execute_percentage_sample_profiling(self):
        """Test percentage-based sampling."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Execute profiling with percentage sampling
            profile = self.use_case.execute_percentage_sample(temp_path, sample_percentage=60.0)
            
            # Verify profile structure
            assert profile is not None
            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.profiling_metadata.sample_percentage == 60.0
            assert profile.profiling_metadata.profiling_strategy == "sample"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_profiling_summary(self):
        """Test profiling summary generation."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Execute profiling
            profile = self.use_case.execute(temp_path, "full")
            
            # Get summary
            summary = self.use_case.get_profiling_summary(profile)
            
            # Verify summary structure
            assert 'profile_id' in summary
            assert 'dataset_id' in summary
            assert 'status' in summary
            assert 'execution_time_seconds' in summary
            assert 'memory_usage_mb' in summary
            assert 'total_rows' in summary
            assert 'total_columns' in summary
            assert 'overall_quality_score' in summary
            assert 'quality_issues' in summary
            assert 'data_types' in summary
            assert 'patterns_discovered' in summary
            
            # Verify values
            assert summary['total_rows'] == 5
            assert summary['total_columns'] == 6
            assert summary['status'] == 'completed'
            assert summary['patterns_discovered'] > 0  # Should find email patterns
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_analyze_relationships(self):
        """Test relationship analysis."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Analyze relationships
            relationships = self.use_case.analyze_relationships(temp_path)
            
            # Verify relationships structure
            assert isinstance(relationships, dict)
            assert 'cross_column_patterns' in relationships
            assert 'functional_dependencies' in relationships
            assert 'categorical_correlations' in relationships
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling_empty_file(self):
        """Test error handling for empty files."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n")  # Just headers
            temp_path = f.name
        
        try:
            # This should raise an error
            with pytest.raises(ValueError, match="Dataset is empty"):
                self.use_case.execute(temp_path, "full")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid file paths."""
        with pytest.raises(Exception):
            self.use_case.execute("nonexistent_file.csv", "full")


class TestPerformanceOptimizer:
    
    def setup_method(self):
        self.optimizer = PerformanceOptimizer()
        
        # Create larger sample data for testing optimization
        self.large_sample_data = pd.DataFrame({
            'id': range(10000),
            'category': ['A', 'B', 'C'] * (10000 // 3) + ['A'],
            'value': [i * 1.5 for i in range(10000)],
            'text': [f'text_{i}' for i in range(10000)]
        })
    
    def test_optimize_profiling_strategy(self):
        """Test profiling strategy optimization."""
        # Test with small dataset
        small_df = self.large_sample_data.head(100)
        strategy = self.optimizer.optimize_profiling_strategy(small_df, {})
        
        assert strategy['strategy'] in ['full_analysis', 'optimized_analysis', 'sampled_analysis', 'chunked_analysis']
        assert 'sampling_config' in strategy
        assert 'parallel_config' in strategy
    
    def test_apply_sampling(self):
        """Test sampling application."""
        # Test systematic sampling
        sampling_config = {'method': 'systematic', 'sample_size': 1000}
        sampled_df = self.optimizer.apply_sampling(self.large_sample_data, sampling_config)
        
        assert len(sampled_df) == 1000
        assert len(sampled_df.columns) == len(self.large_sample_data.columns)
        
        # Test adaptive sampling
        sampling_config = {'method': 'adaptive', 'target_size_mb': 1.0}
        sampled_df = self.optimizer.apply_sampling(self.large_sample_data, sampling_config)
        
        assert len(sampled_df) <= len(self.large_sample_data)
    
    def test_check_system_resources(self):
        """Test system resource checking."""
        resources = self.optimizer.check_system_resources()
        
        assert 'memory' in resources
        assert 'cpu' in resources
        assert 'recommendations' in resources
        
        assert 'total_gb' in resources['memory']
        assert 'available_gb' in resources['memory']
        assert 'used_percent' in resources['memory']
        
        assert 'count' in resources['cpu']
        assert 'usage_percent' in resources['cpu']


class TestSamplingStrategy:
    
    def setup_method(self):
        self.test_data = pd.DataFrame({
            'id': range(1000),
            'category': ['A', 'B', 'C'] * (1000 // 3) + ['A'],
            'value': [i * 0.5 for i in range(1000)]
        })
    
    def test_systematic_sampling(self):
        """Test systematic sampling."""
        sample_size = 100
        sampled_df = SamplingStrategy.systematic_sampling(self.test_data, sample_size)
        
        assert len(sampled_df) == sample_size
        assert len(sampled_df.columns) == len(self.test_data.columns)
        
        # Test with sample size larger than data
        sampled_df = SamplingStrategy.systematic_sampling(self.test_data, 2000)
        assert len(sampled_df) == len(self.test_data)
    
    def test_stratified_sampling(self):
        """Test stratified sampling."""
        sample_size = 150
        sampled_df = SamplingStrategy.stratified_sampling(
            self.test_data, sample_size, 'category'
        )
        
        assert len(sampled_df) <= sample_size
        assert len(sampled_df.columns) == len(self.test_data.columns)
        
        # Verify stratification maintained proportions roughly
        original_proportions = self.test_data['category'].value_counts(normalize=True)
        sample_proportions = sampled_df['category'].value_counts(normalize=True)
        
        for category in original_proportions.index:
            if category in sample_proportions.index:
                # Allow some variance in proportions
                assert abs(original_proportions[category] - sample_proportions[category]) < 0.3
    
    def test_reservoir_sampling(self):
        """Test reservoir sampling."""
        sample_size = 100
        sampled_df = SamplingStrategy.reservoir_sampling(self.test_data, sample_size)
        
        assert len(sampled_df) == sample_size
        assert len(sampled_df.columns) == len(self.test_data.columns)
    
    def test_adaptive_sampling(self):
        """Test adaptive sampling."""
        target_size_mb = 0.1  # Very small target
        sampled_df = SamplingStrategy.adaptive_sampling(self.test_data, target_size_mb)
        
        # Should be smaller than original
        assert len(sampled_df) <= len(self.test_data)
        
        # Memory usage should be close to target
        actual_size_mb = sampled_df.memory_usage(deep=True).sum() / (1024 * 1024)
        assert actual_size_mb <= target_size_mb * 1.5  # Allow some variance


class TestFileAdapters:
    
    def test_csv_adapter(self):
        """Test CSV file adapter."""
        sample_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            adapter = get_file_adapter(temp_path)
            loaded_df = adapter.load(temp_path)
            
            assert len(loaded_df) == 3
            assert len(loaded_df.columns) == 2
            assert 'col1' in loaded_df.columns
            assert 'col2' in loaded_df.columns
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_json_adapter(self):
        """Test JSON file adapter."""
        sample_data = [
            {'col1': 1, 'col2': 'a'},
            {'col1': 2, 'col2': 'b'},
            {'col1': 3, 'col2': 'c'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(sample_data, f)
            temp_path = f.name
        
        try:
            adapter = get_file_adapter(temp_path)
            loaded_df = adapter.load(temp_path)
            
            assert len(loaded_df) == 3
            assert len(loaded_df.columns) == 2
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_unsupported_format(self):
        """Test unsupported file format handling."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            get_file_adapter("test.xyz")


# Legacy test for backwards compatibility
@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        'a': [1, 2, 2, None],
        'b': ['x', 'y', 'y', 'z'],
        'c': ['test@example.com', 'foo@bar.com', 'not_email', 'hello@example.org']
    })
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    return str(file)

def test_profile_csv(csv_file):
    use_case = ProfileDatasetUseCase()
    profile = use_case.execute(csv_file)
    schema = profile.schema_profile
    # Check basic schema info
    assert schema.total_columns == 3
    col_names = [col.column_name for col in schema.columns]
    assert set(col_names) == {'a', 'b', 'c'}
    
    # Check for email patterns in column 'c'
    email_column = next((col for col in schema.columns if col.column_name == 'c'), None)
    assert email_column is not None
    assert len(email_column.patterns) > 0  # Should detect email pattern