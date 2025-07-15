import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from ..application.services.schema_analysis_service import SchemaAnalysisService
from ..domain.entities.data_profile import DataType, CardinalityLevel, QualityIssueType


class TestSchemaAnalysisService:
    
    def setup_method(self):
        self.service = SchemaAnalysisService()
    
    def test_infer_basic_schema(self):
        """Test basic schema inference."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 22],
            'score': [85.5, 92.0, 78.5, 88.0, 91.5]
        })
        
        schema_profile = self.service.infer(df)
        
        assert schema_profile.table_name == "dataset"
        assert schema_profile.total_columns == 4
        assert schema_profile.total_rows == 5
        assert len(schema_profile.columns) == 4
        
        # Check column types
        column_dict = {col.column_name: col for col in schema_profile.columns}
        
        assert column_dict['id'].data_type == DataType.INTEGER
        assert column_dict['name'].data_type == DataType.STRING
        assert column_dict['age'].data_type == DataType.INTEGER
        assert column_dict['score'].data_type == DataType.FLOAT
    
    def test_cardinality_calculation(self):
        """Test cardinality level calculation."""
        # Low cardinality
        assert self.service._calculate_cardinality(5, 100) == CardinalityLevel.LOW
        
        # Medium cardinality
        assert self.service._calculate_cardinality(50, 100) == CardinalityLevel.MEDIUM
        
        # High cardinality
        assert self.service._calculate_cardinality(500, 1000) == CardinalityLevel.HIGH
        
        # Very high cardinality
        assert self.service._calculate_cardinality(5000, 5000) == CardinalityLevel.VERY_HIGH
    
    def test_missing_values_quality_assessment(self):
        """Test quality assessment for missing values."""
        df = pd.DataFrame({
            'col_with_nulls': [1, 2, None, 4, None],
            'col_without_nulls': [1, 2, 3, 4, 5]
        })
        
        schema_profile = self.service.infer(df)
        
        # Check column with nulls
        col_with_nulls = next(col for col in schema_profile.columns if col.column_name == 'col_with_nulls')
        assert col_with_nulls.distribution.null_count == 2
        assert col_with_nulls.distribution.completeness_ratio == 0.6
        assert len(col_with_nulls.quality_issues) > 0
        assert any(issue.issue_type == QualityIssueType.MISSING_VALUES for issue in col_with_nulls.quality_issues)
        
        # Check column without nulls
        col_without_nulls = next(col for col in schema_profile.columns if col.column_name == 'col_without_nulls')
        assert col_without_nulls.distribution.null_count == 0
        assert col_without_nulls.distribution.completeness_ratio == 1.0
    
    def test_outlier_detection(self):
        """Test outlier detection in quality assessment."""
        # Create data with clear outliers
        normal_data = np.random.normal(50, 10, 95)
        outliers = [150, 200]  # Clear outliers
        data = np.concatenate([normal_data, outliers])
        
        df = pd.DataFrame({'values': data})
        schema_profile = self.service.infer(df)
        
        col_profile = schema_profile.columns[0]
        outlier_issues = [issue for issue in col_profile.quality_issues 
                         if issue.issue_type == QualityIssueType.OUTLIERS]
        
        assert len(outlier_issues) > 0
        assert outlier_issues[0].affected_rows > 0
    
    def test_primary_key_detection(self):
        """Test primary key detection."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],  # Unique values, potential PK
            'name': ['A', 'B', 'C', 'D', 'E'],  # Unique values, potential PK
            'category': ['X', 'X', 'Y', 'Y', 'Z']  # Not unique
        })
        
        schema_profile = self.service.infer(df)
        
        # Both id and name should be detected as potential primary keys
        assert 'id' in schema_profile.primary_keys
        assert 'name' in schema_profile.primary_keys
        assert 'category' not in schema_profile.primary_keys
    
    def test_unique_constraints_detection(self):
        """Test unique constraint detection."""
        df = pd.DataFrame({
            'unique_col': [1, 2, 3, 4, 5],
            'non_unique_col': [1, 1, 2, 2, 3]
        })
        
        schema_profile = self.service.infer(df)
        
        # Should detect unique constraint on unique_col
        unique_constraints = schema_profile.unique_constraints
        assert any('unique_col' in constraint for constraint in unique_constraints)
        assert not any('non_unique_col' in constraint for constraint in unique_constraints)
    
    def test_statistical_summary_creation(self):
        """Test statistical summary creation for numeric columns."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        schema_profile = self.service.infer(df)
        col_profile = schema_profile.columns[0]
        
        assert col_profile.statistical_summary is not None
        assert col_profile.statistical_summary.min_value == 1.0
        assert col_profile.statistical_summary.max_value == 10.0
        assert col_profile.statistical_summary.mean == 5.5
        assert col_profile.statistical_summary.median == 5.5
        assert len(col_profile.statistical_summary.quartiles) == 3
    
    def test_data_type_inference(self):
        """Test data type inference for various pandas dtypes."""
        # Test various data types
        assert self.service._infer_data_type(pd.Series([1, 2, 3]), 'int64') == DataType.INTEGER
        assert self.service._infer_data_type(pd.Series([1.1, 2.2, 3.3]), 'float64') == DataType.FLOAT
        assert self.service._infer_data_type(pd.Series([True, False, True]), 'bool') == DataType.BOOLEAN
        assert self.service._infer_data_type(pd.Series(['a', 'b', 'c']), 'object') == DataType.STRING
        
        # Test datetime inference
        dates = pd.Series([datetime.now(), datetime.now()])
        assert self.service._infer_data_type(dates, 'datetime64[ns]') == DataType.DATETIME
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        schema_profile = self.service.infer(df)
        
        assert schema_profile.total_columns == 0
        assert schema_profile.total_rows == 0
        assert len(schema_profile.columns) == 0
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a larger dataset
        df = pd.DataFrame({
            'id': range(10000),
            'category': ['A', 'B', 'C'] * 3334,  # Repeating pattern
            'value': np.random.normal(100, 15, 10000)
        })
        
        schema_profile = self.service.infer(df)
        
        assert schema_profile.total_rows == 10000
        assert schema_profile.total_columns == 3
        
        # Check that analysis completes without errors
        id_col = next(col for col in schema_profile.columns if col.column_name == 'id')
        assert id_col.cardinality == CardinalityLevel.VERY_HIGH
        
        category_col = next(col for col in schema_profile.columns if col.column_name == 'category')
        assert category_col.cardinality == CardinalityLevel.LOW
    
    def test_mixed_type_column(self):
        """Test handling of columns with mixed types."""
        df = pd.DataFrame({
            'mixed_col': [1, '2', 3.0, 'four', 5]
        })
        
        schema_profile = self.service.infer(df)
        col_profile = schema_profile.columns[0]
        
        # Should be inferred as string type due to mixed content
        assert col_profile.data_type == DataType.STRING
    
    def test_duplicate_detection(self):
        """Test duplicate value detection."""
        df = pd.DataFrame({
            'with_duplicates': [1, 2, 2, 3, 3, 3],
            'without_duplicates': [1, 2, 3, 4, 5, 6]
        })
        
        schema_profile = self.service.infer(df)
        
        # Check column with duplicates
        dup_col = next(col for col in schema_profile.columns if col.column_name == 'with_duplicates')
        duplicate_issues = [issue for issue in dup_col.quality_issues 
                          if issue.issue_type == QualityIssueType.DUPLICATE_VALUES]
        assert len(duplicate_issues) > 0
        
        # Check column without duplicates
        no_dup_col = next(col for col in schema_profile.columns if col.column_name == 'without_duplicates')
        duplicate_issues = [issue for issue in no_dup_col.quality_issues 
                          if issue.issue_type == QualityIssueType.DUPLICATE_VALUES]
        assert len(duplicate_issues) == 0
    
    @pytest.mark.parametrize("data,expected_type", [
        ([1, 2, 3], DataType.INTEGER),
        ([1.1, 2.2, 3.3], DataType.FLOAT),
        (['a', 'b', 'c'], DataType.STRING),
        ([True, False, True], DataType.BOOLEAN),
    ])
    def test_type_inference_parametrized(self, data, expected_type):
        """Parametrized test for type inference."""
        df = pd.DataFrame({'col': data})
        schema_profile = self.service.infer(df)
        
        assert schema_profile.columns[0].data_type == expected_type