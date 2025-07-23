"""Comprehensive test suite for DataValidationService."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from anomaly_detection.domain.services.data_validation_service import DataValidationService


class TestDataValidationService:
    """Test suite for DataValidationService."""
    
    @pytest.fixture
    def validation_service(self):
        """Create DataValidationService instance."""
        return DataValidationService()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.uniform(-5, 5, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid test data with various issues."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, np.nan] * 40,  # 40% missing
            'feature2': [1, 1, 1, 1, 1] * 40,  # All same values
            'feature3': [1, 2, 3, 999999, 5] * 40,  # Contains outliers
        })
        # Add duplicates
        data = pd.concat([data, data.iloc[:50]], ignore_index=True)
        return data
    
    @pytest.fixture
    def temp_csv_file(self, valid_data):
        """Create temporary CSV file with valid data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            valid_data.to_csv(f.name, index=False)
            yield Path(f.name)
            Path(f.name).unlink()
    
    @pytest.fixture
    def temp_invalid_csv_file(self, invalid_data):
        """Create temporary CSV file with invalid data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            yield Path(f.name)
            Path(f.name).unlink()
    
    @pytest.fixture
    def sample_schema(self):
        """Create sample JSON schema for validation."""
        return {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "number"},
                "feature3": {"type": "number"},
                "category": {"type": "string", "enum": ["A", "B", "C"]}
            },
            "required": ["feature1", "feature2"]
        }
    
    @pytest.fixture
    def temp_schema_file(self, sample_schema):
        """Create temporary schema file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_schema, f)
            yield Path(f.name)
            Path(f.name).unlink()

    def test_init_default_rules(self, validation_service):
        """Test initialization with default validation rules."""
        assert validation_service.validation_rules is not None
        assert 'max_missing_percentage' in validation_service.validation_rules
        assert 'max_duplicate_percentage' in validation_service.validation_rules
        assert 'min_numeric_columns' in validation_service.validation_rules
        assert 'max_outlier_percentage' in validation_service.validation_rules
        assert 'required_row_count' in validation_service.validation_rules

    def test_load_default_rules(self, validation_service):
        """Test loading of default validation rules."""
        rules = validation_service._load_default_rules()
        
        assert rules['max_missing_percentage'] == 50.0
        assert rules['max_duplicate_percentage'] == 20.0
        assert rules['min_numeric_columns'] == 1
        assert rules['max_outlier_percentage'] == 10.0
        assert rules['required_row_count'] == 10

    @pytest.mark.asyncio
    async def test_validate_file_valid_data(self, validation_service, temp_csv_file):
        """Test validation of valid data file."""
        result = await validation_service.validate_file(temp_csv_file)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert 'statistics' in result
        assert 'file_path' in result
        assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_validate_file_invalid_data(self, validation_service, temp_invalid_csv_file):
        """Test validation of invalid data file."""
        result = await validation_service.validate_file(temp_invalid_csv_file)
        
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        # Should detect missing values and duplicates
        error_messages = ' '.join(result['errors'])
        assert 'missing' in error_messages.lower() or 'duplicate' in error_messages.lower()

    @pytest.mark.asyncio
    async def test_validate_file_nonexistent(self, validation_service):
        """Test validation of nonexistent file."""
        nonexistent_file = Path("nonexistent.csv")
        
        result = await validation_service.validate_file(nonexistent_file)
        
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        assert 'not found' in result['errors'][0].lower() or 'does not exist' in result['errors'][0].lower()

    @pytest.mark.asyncio
    async def test_validate_file_with_schema(self, validation_service, temp_csv_file, temp_schema_file):
        """Test validation with JSON schema."""
        result = await validation_service.validate_file(
            temp_csv_file, 
            schema_file=temp_schema_file
        )
        
        assert 'schema_validation' in result
        # Should pass schema validation for valid data
        assert result['is_valid'] is True

    @pytest.mark.asyncio
    async def test_validate_file_custom_rules(self, validation_service, temp_csv_file):
        """Test validation with custom rules."""
        custom_rules = {
            'max_missing_percentage': 5.0,  # Very strict
            'required_row_count': 500
        }
        
        result = await validation_service.validate_file(
            temp_csv_file,
            custom_rules=custom_rules
        )
        
        # Should use custom rules for validation
        assert result is not None

    @pytest.mark.asyncio
    async def test_validate_file_selective_checks(self, validation_service, temp_csv_file):
        """Test validation with selective checks disabled."""
        result = await validation_service.validate_file(
            temp_csv_file,
            check_types=False,
            check_missing=False,
            check_outliers=False,
            check_duplicates=False
        )
        
        # Should still return result but with fewer checks
        assert result['is_valid'] is not None
        assert 'statistics' in result

    def test_check_missing_values_clean_data(self, validation_service, valid_data):
        """Test missing value check on clean data."""
        issues = validation_service._check_missing_values(valid_data)
        
        assert len(issues) == 0

    def test_check_missing_values_with_missing(self, validation_service):
        """Test missing value check on data with missing values."""
        data_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, np.nan],
            'col2': [1, np.nan, 3, np.nan, 5]
        })
        
        issues = validation_service._check_missing_values(data_with_missing)
        
        assert len(issues) > 0
        assert any('missing' in issue.lower() for issue in issues)

    def test_check_duplicates_clean_data(self, validation_service, valid_data):
        """Test duplicate check on clean data."""
        issues = validation_service._check_duplicates(valid_data)
        
        # Random data should have very few or no duplicates
        assert len(issues) == 0 or 'low' in issues[0].lower()

    def test_check_duplicates_with_duplicates(self, validation_service):
        """Test duplicate check on data with many duplicates."""
        data_with_duplicates = pd.DataFrame({
            'col1': [1, 1, 1, 2, 2],
            'col2': [1, 1, 1, 2, 2]
        })
        
        issues = validation_service._check_duplicates(data_with_duplicates)
        
        assert len(issues) > 0
        assert any('duplicate' in issue.lower() for issue in issues)

    def test_check_data_types_numeric_data(self, validation_service, valid_data):
        """Test data type check on mostly numeric data."""
        issues = validation_service._check_data_types(valid_data)
        
        # Should pass minimum numeric columns requirement
        assert len(issues) == 0

    def test_check_data_types_insufficient_numeric(self, validation_service):
        """Test data type check with insufficient numeric columns."""
        text_data = pd.DataFrame({
            'text1': ['a', 'b', 'c'],
            'text2': ['x', 'y', 'z']
        })
        
        issues = validation_service._check_data_types(text_data)
        
        assert len(issues) > 0
        assert any('numeric' in issue.lower() for issue in issues)

    def test_check_outliers_clean_data(self, validation_service):
        """Test outlier detection on clean data."""
        clean_data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 1000)
        })
        
        issues = validation_service._check_outliers(clean_data)
        
        # Should have few outliers in normal distribution
        assert len(issues) == 0 or 'acceptable' in issues[0].lower()

    def test_check_outliers_with_outliers(self, validation_service):
        """Test outlier detection on data with clear outliers."""
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 1000, -1000, 2000]  # Clear outliers
        })
        
        issues = validation_service._check_outliers(data_with_outliers)
        
        assert len(issues) > 0
        assert any('outlier' in issue.lower() for issue in issues)

    def test_check_row_count_sufficient(self, validation_service, valid_data):
        """Test row count check with sufficient data."""
        issues = validation_service._check_row_count(valid_data)
        
        assert len(issues) == 0

    def test_check_row_count_insufficient(self, validation_service):
        """Test row count check with insufficient data."""
        small_data = pd.DataFrame({'col1': [1, 2, 3]})  # Only 3 rows
        
        issues = validation_service._check_row_count(small_data)
        
        assert len(issues) > 0
        assert any('row' in issue.lower() or 'sample' in issue.lower() for issue in issues)

    def test_validate_against_schema_valid(self, validation_service, sample_schema):
        """Test schema validation with valid data."""
        valid_record = {
            'feature1': 1.5,
            'feature2': 2.5,
            'feature3': 3.5,
            'category': 'A'
        }
        
        is_valid, errors = validation_service._validate_against_schema(valid_record, sample_schema)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_against_schema_invalid(self, validation_service, sample_schema):
        """Test schema validation with invalid data."""
        invalid_record = {
            'feature1': 'not_a_number',  # Should be number
            'feature2': 2.5,
            'category': 'INVALID'  # Not in enum
        }
        
        is_valid, errors = validation_service._validate_against_schema(invalid_record, sample_schema)
        
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_against_schema_missing_required(self, validation_service, sample_schema):
        """Test schema validation with missing required fields."""
        incomplete_record = {
            'feature3': 3.5,
            'category': 'B'
            # Missing required feature1 and feature2
        }
        
        is_valid, errors = validation_service._validate_against_schema(incomplete_record, sample_schema)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any('required' in error.lower() for error in errors)

    def test_compute_statistics(self, validation_service, valid_data):
        """Test computation of data statistics."""
        stats = validation_service._compute_statistics(valid_data)
        
        assert 'row_count' in stats
        assert 'column_count' in stats
        assert 'numeric_columns' in stats
        assert 'missing_percentage' in stats
        assert 'duplicate_percentage' in stats
        
        assert stats['row_count'] == len(valid_data)
        assert stats['column_count'] == len(valid_data.columns)

    @pytest.mark.asyncio
    async def test_validate_multiple_files(self, validation_service, valid_data):
        """Test validation of multiple files."""
        # Create multiple temporary files
        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.csv', delete=False) as f:
                    valid_data.to_csv(f.name, index=False)
                    temp_files.append(Path(f.name))
            
            results = await validation_service.validate_multiple_files(temp_files)
            
            assert len(results) == 3
            for result in results:
                assert 'is_valid' in result
                assert 'file_path' in result
                
        finally:
            # Cleanup
            for temp_file in temp_files:
                temp_file.unlink()

    @pytest.mark.asyncio
    async def test_validate_directory(self, validation_service, valid_data):
        """Test validation of entire directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            for i in range(3):
                file_path = temp_path / f"data_{i}.csv"
                valid_data.to_csv(file_path, index=False)
            
            results = await validation_service.validate_directory(
                temp_path, 
                pattern="*.csv"
            )
            
            assert len(results) >= 3
            for result in results:
                assert result['is_valid'] is not None

    @pytest.mark.asyncio
    async def test_validate_streaming_data(self, validation_service):
        """Test validation of streaming data chunks."""
        # Simulate streaming data
        data_chunks = [
            pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
            pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]}),
            pd.DataFrame({'col1': [13, np.nan, 15], 'col2': [16, 17, 18]})  # Contains missing
        ]
        
        results = []
        for chunk in data_chunks:
            result = await validation_service.validate_streaming_chunk(chunk)
            results.append(result)
        
        assert len(results) == 3
        # Last chunk should have validation issues due to missing value
        assert results[-1]['is_valid'] is False

    def test_create_validation_report(self, validation_service):
        """Test creation of validation report."""
        validation_results = [
            {'file_path': 'file1.csv', 'is_valid': True, 'errors': [], 'warnings': []},
            {'file_path': 'file2.csv', 'is_valid': False, 'errors': ['Missing values'], 'warnings': ['Duplicates']},
            {'file_path': 'file3.csv', 'is_valid': True, 'errors': [], 'warnings': ['Minor issues']}
        ]
        
        report = validation_service.create_validation_report(validation_results)
        
        assert 'summary' in report
        assert 'total_files' in report['summary']
        assert 'valid_files' in report['summary']
        assert 'invalid_files' in report['summary']
        assert 'details' in report
        
        assert report['summary']['total_files'] == 3
        assert report['summary']['valid_files'] == 2
        assert report['summary']['invalid_files'] == 1

    def test_get_validation_recommendations(self, validation_service):
        """Test generation of validation recommendations."""
        validation_issues = [
            'High percentage of missing values in column feature1',
            'Excessive duplicate rows detected',
            'Outliers detected in column feature2'
        ]
        
        recommendations = validation_service.get_validation_recommendations(validation_issues)
        
        assert len(recommendations) > 0
        assert any('missing' in rec.lower() for rec in recommendations)
        assert any('duplicate' in rec.lower() for rec in recommendations)
        assert any('outlier' in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_performance_large_file(self, validation_service):
        """Test validation performance on large file."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': np.random.rand(10000),
            'col2': np.random.rand(10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_file = Path(f.name)
        
        try:
            import time
            start_time = time.time()
            
            result = await validation_service.validate_file(temp_file)
            
            processing_time = time.time() - start_time
            
            assert result is not None
            assert processing_time < 30  # Should complete within reasonable time
            
        finally:
            temp_file.unlink()

    def test_custom_validation_rule(self, validation_service):
        """Test application of custom validation rules."""
        custom_rules = {
            'max_missing_percentage': 1.0,  # Very strict
            'min_unique_values_per_column': 5
        }
        
        validation_service.validation_rules.update(custom_rules)
        
        assert validation_service.validation_rules['max_missing_percentage'] == 1.0
        assert validation_service.validation_rules['min_unique_values_per_column'] == 5

    def test_error_handling_corrupted_file(self, validation_service):
        """Test error handling for corrupted files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("corrupted,csv\ndata,that,has\ntoo,many,columns,here")
            corrupted_file = Path(f.name)
        
        try:
            # This should be tested with asyncio.run in actual test
            import asyncio
            result = asyncio.run(validation_service.validate_file(corrupted_file))
            
            assert result['is_valid'] is False
            assert len(result['errors']) > 0
            
        finally:
            corrupted_file.unlink()

    def teardown_method(self):
        """Cleanup after each test."""
        pass