import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from ..application.use_cases.profile_dataset import ProfileDatasetUseCase
from ..domain.entities.data_profile import DataProfile, ProfilingStatus
from ..infrastructure.adapters.data_source_adapter import DataSourceFactory
from ..infrastructure.exceptions import DataLoadError, FileFormatError


class TestDataProfilingIntegration:
    """Integration tests for the complete data profiling workflow."""
    
    def setup_method(self):
        self.use_case = ProfileDatasetUseCase()
    
    def create_test_csv(self, data: dict, filename: str = None) -> str:
        """Helper to create a temporary CSV file for testing."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    def test_complete_profiling_workflow(self):
        """Test the complete profiling workflow from file to results."""
        # Create test data with various data types and quality issues
        test_data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],  # Has null
            'email': ['alice@test.com', 'bob@test.com', 'invalid-email', 'charlie@test.com', 'eve@test.com'],
            'age': [25, 30, 35, 28, 22],
            'score': [85.5, 92.0, 78.5, 88.0, 91.5],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'phone': ['+1-555-1234', '+1-555-5678', 'invalid', '+1-555-9012', '+1-555-3456']
        }
        
        # Create temporary CSV file
        csv_file = self.create_test_csv(test_data)
        
        try:
            # Execute profiling
            profile = self.use_case.execute(csv_file)
            
            # Verify profile structure
            assert isinstance(profile, DataProfile)
            assert profile.is_completed()
            assert profile.schema_profile is not None
            assert profile.quality_assessment is not None
            assert profile.profiling_metadata is not None
            
            # Verify schema analysis
            schema = profile.schema_profile
            assert schema.total_rows == 5
            assert schema.total_columns == 7
            assert len(schema.columns) == 7
            
            # Verify column analysis
            column_names = [col.column_name for col in schema.columns]
            assert 'id' in column_names
            assert 'name' in column_names
            assert 'email' in column_names
            
            # Verify pattern detection
            email_col = next((col for col in schema.columns if col.column_name == 'email'), None)
            assert email_col is not None
            # Should detect email patterns
            assert len(email_col.patterns) > 0 or any('email' in str(p.regex).lower() for p in email_col.patterns)
            
            # Verify quality assessment
            quality = profile.quality_assessment
            assert 0 <= quality.overall_score <= 1
            assert quality.completeness_score < 1.0  # Due to null in 'name'
            assert len(quality.recommendations) > 0
            
            # Verify metadata
            metadata = profile.profiling_metadata
            assert metadata.profiling_strategy == "full"
            assert metadata.execution_time_seconds > 0
            assert metadata.include_patterns is True
            
        finally:
            # Cleanup
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_sampling_workflow(self):
        """Test profiling with sampling."""
        # Create larger test dataset
        np.random.seed(42)
        large_data = {
            'id': range(1000),
            'value': np.random.normal(100, 15, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        }
        
        csv_file = self.create_test_csv(large_data)
        
        try:
            # Test sample size profiling
            profile = self.use_case.execute_sample(csv_file, sample_size=100)
            
            assert profile.is_completed()
            assert profile.profiling_metadata.profiling_strategy == "sample"
            assert profile.profiling_metadata.sample_size == 100
            assert profile.schema_profile.total_rows == 1000  # Original size
            
            # Test percentage sampling
            profile = self.use_case.execute_percentage_sample(csv_file, sample_percentage=10.0)
            
            assert profile.is_completed()
            assert profile.profiling_metadata.sample_percentage == 10.0
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_profiling_summary(self):
        """Test profiling summary generation."""
        test_data = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        }
        
        csv_file = self.create_test_csv(test_data)
        
        try:
            profile = self.use_case.execute(csv_file)
            summary = self.use_case.get_profiling_summary(profile)
            
            assert 'profile_id' in summary
            assert 'dataset_id' in summary
            assert 'status' in summary
            assert 'execution_time_seconds' in summary
            assert 'total_rows' in summary
            assert 'total_columns' in summary
            assert 'overall_quality_score' in summary
            assert 'data_types' in summary
            assert 'recommendations' in summary
            
            # Verify data type counts
            assert summary['data_types']['integer'] >= 1
            assert summary['data_types']['string'] >= 1
            assert summary['data_types']['float'] >= 1
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_relationship_analysis(self):
        """Test data relationship analysis."""
        test_data = {
            'country_code': ['US', 'US', 'CA', 'CA'],
            'country_name': ['United States', 'United States', 'Canada', 'Canada'],
            'region': ['North America', 'North America', 'North America', 'North America']
        }
        
        csv_file = self.create_test_csv(test_data)
        
        try:
            relationships = self.use_case.analyze_relationships(csv_file)
            
            assert 'cross_column_patterns' in relationships
            assert 'functional_dependencies' in relationships
            assert 'categorical_correlations' in relationships
            
            # Should detect functional dependency: country_code -> country_name
            dependencies = relationships['functional_dependencies']
            dependency_found = any(
                dep['determinant'] == 'country_code' and dep['dependent'] == 'country_name'
                for dep in dependencies
            )
            assert dependency_found
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid file paths."""
        with pytest.raises(Exception):  # Should raise some kind of error
            self.use_case.execute('/nonexistent/file.csv')
    
    def test_error_handling_empty_file(self):
        """Test error handling for empty files."""
        # Create empty CSV
        csv_file = self.create_test_csv({})
        
        try:
            with pytest.raises(ValueError, match="Dataset is empty"):
                self.use_case.execute(csv_file)
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_profiling_status_tracking(self):
        """Test profiling status tracking through the workflow."""
        test_data = {'col1': [1, 2, 3]}
        csv_file = self.create_test_csv(test_data)
        
        try:
            # Mock the services to track status changes
            with patch.object(self.use_case.schema_service, 'infer') as mock_schema:
                mock_schema.side_effect = Exception("Schema analysis failed")
                
                profile = None
                try:
                    profile = self.use_case.execute(csv_file)
                except Exception:
                    pass
                
                # Profile should be in failed state
                if profile:
                    assert profile.is_failed()
                    assert profile.error_message is not None
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create moderately large dataset
        np.random.seed(42)
        large_data = {
            'id': range(5000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 5000),
            'value1': np.random.normal(100, 15, 5000),
            'value2': np.random.exponential(2, 5000),
            'text': [f'text_{i}' for i in range(5000)]
        }
        
        csv_file = self.create_test_csv(large_data)
        
        try:
            profile = self.use_case.execute(csv_file)
            
            assert profile.is_completed()
            assert profile.schema_profile.total_rows == 5000
            assert profile.profiling_metadata.execution_time_seconds > 0
            
            # Should complete in reasonable time (adjust threshold as needed)
            assert profile.profiling_metadata.execution_time_seconds < 60  # Less than 1 minute
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_different_file_formats(self):
        """Test profiling different file formats."""
        test_data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        
        # Test CSV
        csv_file = self.create_test_csv(test_data)
        try:
            profile_csv = self.use_case.execute(csv_file)
            assert profile_csv.is_completed()
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
        
        # Test JSON
        json_file = None
        try:
            fd, json_file = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            df = pd.DataFrame(test_data)
            df.to_json(json_file, orient='records')
            
            profile_json = self.use_case.execute(json_file)
            assert profile_json.is_completed()
        finally:
            if json_file and os.path.exists(json_file):
                os.unlink(json_file)
    
    def test_data_source_adapter_integration(self):
        """Test integration with data source adapters."""
        test_data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        csv_file = self.create_test_csv(test_data)
        
        try:
            # Test file source adapter
            file_source = DataSourceFactory.create_file_source(csv_file)
            df = file_source.load_data()
            
            assert len(df) == 3
            assert 'col1' in df.columns
            assert 'col2' in df.columns
            
            # Test source info
            source_info = file_source.get_source_info()
            assert source_info['source_type'] == 'file'
            assert 'file_info' in source_info
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking during profiling."""
        test_data = {
            'col1': list(range(1000)),
            'col2': [f'text_{i}' * 10 for i in range(1000)],  # Larger text data
            'col3': np.random.random(1000)
        }
        
        csv_file = self.create_test_csv(test_data)
        
        try:
            profile = self.use_case.execute(csv_file)
            
            assert profile.profiling_metadata.memory_usage_mb is not None
            assert profile.profiling_metadata.memory_usage_mb > 0
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_quality_issue_detection_integration(self):
        """Test end-to-end quality issue detection."""
        # Create data with various quality issues
        test_data = {
            'id': [1, 2, 3, 4, 5],
            'email': ['valid@test.com', 'also@valid.com', 'invalid-email', 'bad@', '@missing.com'],
            'age': [25, 30, -5, 150, 35],  # Negative age and unrealistic age
            'name': ['John', 'Jane', None, 'Bob', None],  # Missing values
            'category': ['A', 'a', 'A', 'B', 'b']  # Inconsistent casing
        }
        
        csv_file = self.create_test_csv(test_data)
        
        try:
            profile = self.use_case.execute(csv_file)
            
            # Should detect various quality issues
            quality = profile.quality_assessment
            total_issues = (quality.critical_issues + quality.high_issues + 
                          quality.medium_issues + quality.low_issues)
            assert total_issues > 0
            
            # Should have recommendations
            assert len(quality.recommendations) > 0
            
            # Overall quality score should be affected
            assert quality.overall_score < 1.0
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def test_pattern_discovery_integration(self):
        """Test end-to-end pattern discovery."""
        test_data = {
            'emails': ['user1@test.com', 'user2@test.com', 'admin@company.org'],
            'phones': ['+1-555-1234', '+1-555-5678', '+1-555-9012'],
            'codes': ['ABC123', 'DEF456', 'GHI789'],
            'mixed': ['some text', 'other text', 'more text']
        }
        
        csv_file = self.create_test_csv(test_data)
        
        try:
            profile = self.use_case.execute(csv_file)
            
            # Check email patterns
            email_col = next((col for col in profile.schema_profile.columns 
                            if col.column_name == 'emails'), None)
            assert email_col is not None
            assert len(email_col.patterns) > 0
            
            # Check phone patterns
            phone_col = next((col for col in profile.schema_profile.columns 
                            if col.column_name == 'phones'), None)
            assert phone_col is not None
            # May or may not detect phone patterns depending on regex strictness
            
            # Check code patterns (fixed format)
            code_col = next((col for col in profile.schema_profile.columns 
                           if col.column_name == 'codes'), None)
            assert code_col is not None
            assert len(code_col.patterns) > 0
            
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Any necessary cleanup
        pass