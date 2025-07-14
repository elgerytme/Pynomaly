import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from scipy import stats

from ..application.services.statistical_profiling_service import StatisticalProfilingService
from ..domain.entities.data_profile import StatisticalSummary


class TestStatisticalProfilingService:
    
    def setup_method(self):
        self.service = StatisticalProfilingService()
    
    def test_basic_statistical_analysis(self):
        """Test basic statistical analysis for numeric columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10.5, 20.3, 30.1, 40.7, 50.2],
            'text_col': ['a', 'b', 'c', 'd', 'e']  # Should be ignored
        })
        
        results = self.service.analyze(df)
        
        # Should only analyze numeric columns
        assert 'col1' in results
        assert 'col2' in results
        assert 'text_col' not in results
        
        # Check col1 statistics
        stats_col1 = results['col1']
        assert isinstance(stats_col1, StatisticalSummary)
        assert stats_col1.min_value == 1.0
        assert stats_col1.max_value == 5.0
        assert stats_col1.mean == 3.0
        assert stats_col1.median == 3.0
        assert len(stats_col1.quartiles) == 3
    
    def test_empty_series_handling(self):
        """Test handling of empty series."""
        empty_series = pd.Series([], dtype=float)
        
        result = self.service._analyze_numeric_column(empty_series)
        
        assert isinstance(result, StatisticalSummary)
        assert result.min_value is None
        assert result.max_value is None
        assert result.mean is None
    
    def test_series_with_nulls(self):
        """Test handling of series with null values."""
        df = pd.DataFrame({
            'with_nulls': [1, 2, None, 4, 5, None]
        })
        
        results = self.service.analyze(df)
        stats_result = results['with_nulls']
        
        # Should calculate statistics ignoring nulls
        assert stats_result.min_value == 1.0
        assert stats_result.max_value == 5.0
        assert stats_result.mean == 3.0  # (1+2+4+5)/4
    
    def test_distribution_analysis(self):
        """Test distribution analysis functionality."""
        # Create normally distributed data
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 1000)
        series = pd.Series(normal_data)
        
        result = self.service.analyze_distribution(series)
        
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'normality_test' in result
        assert 'best_distribution' in result
        assert 'is_normal' in result
        
        # For normal data, should be close to normal distribution
        assert abs(result['skewness']) < 0.5  # Should be close to 0 for normal data
        assert result['normality_test']['test_name'] in ['Shapiro-Wilk', 'Anderson-Darling']
    
    def test_normality_testing(self):
        """Test normality testing with known distributions."""
        # Normal data
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        result = self.service._test_normality(normal_data)
        
        assert 'test_name' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
        assert 'interpretation' in result
        
        # Uniform data (should not be normal)
        uniform_data = pd.Series(np.random.uniform(0, 1, 100))
        result = self.service._test_normality(uniform_data)
        
        # For uniform data, normality test should likely fail
        assert isinstance(result['is_normal'], bool)
    
    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection."""
        # Create data with known outliers
        normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        outliers = [100, 200]  # Clear outliers
        data = pd.Series(normal_data + outliers)
        
        result = self.service.detect_outliers(data, method='iqr')
        
        assert result['method'] == 'IQR'
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'outlier_count' in result
        assert 'outlier_percentage' in result
        assert 'outlier_values' in result
        
        # Should detect the outliers
        assert result['outlier_count'] > 0
        assert 100 in result['outlier_values'] or 200 in result['outlier_values']
    
    def test_outlier_detection_zscore(self):
        """Test Z-score based outlier detection."""
        # Create data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = [5, -5]  # Z-score > 3
        data = pd.Series(list(normal_data) + outliers)
        
        result = self.service.detect_outliers(data, method='zscore')
        
        assert result['method'] == 'Z-Score'
        assert 'threshold' in result
        assert 'outlier_count' in result
        assert 'max_zscore' in result
        
        # Should detect outliers with |z| > 3
        assert result['outlier_count'] > 0
    
    @patch('sklearn.ensemble.IsolationForest')
    def test_outlier_detection_isolation_forest(self, mock_isolation_forest):
        """Test Isolation Forest outlier detection."""
        # Mock the IsolationForest
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([1, 1, 1, -1, 1])  # One outlier
        mock_isolation_forest.return_value = mock_model
        
        data = pd.Series([1, 2, 3, 100, 5])  # 100 is an outlier
        
        result = self.service.detect_outliers(data, method='isolation_forest')
        
        assert result['method'] == 'Isolation Forest'
        assert 'outlier_count' in result
        assert result['outlier_count'] == 1
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 0.8 * x + 0.2 * np.random.normal(0, 1, 100)  # Strong correlation
        z = np.random.normal(0, 1, 100)  # No correlation
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
        result = self.service.correlation_analysis(df)
        
        assert 'correlation_matrix' in result
        assert 'high_correlations' in result
        assert 'max_correlation' in result
        assert 'min_correlation' in result
        
        # Should detect high correlation between x and y
        high_corrs = result['high_correlations']
        assert len(high_corrs) > 0
        
        # Check if x-y correlation is detected
        xy_correlation = None
        for corr in high_corrs:
            if (corr['column1'] == 'x' and corr['column2'] == 'y') or \
               (corr['column1'] == 'y' and corr['column2'] == 'x'):
                xy_correlation = corr['correlation']
                break
        
        assert xy_correlation is not None
        assert abs(xy_correlation) > 0.7
    
    def test_distribution_fitting(self):
        """Test distribution fitting functionality."""
        # Create data from known distribution
        np.random.seed(42)
        exponential_data = pd.Series(np.random.exponential(2, 1000))
        
        result = self.service._fit_distribution(exponential_data)
        
        assert 'distribution' in result
        assert 'parameters' in result
        assert 'ks_statistic' in result
        assert 'p_value' in result
        assert 'goodness_of_fit' in result
        
        # Should identify some distribution
        assert result['distribution'] != 'Unknown'
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'col1': np.random.normal(100, 15, 10000),
            'col2': np.random.uniform(0, 1, 10000),
            'col3': np.random.exponential(2, 10000)
        })
        
        # Should complete without errors
        results = self.service.analyze(large_df)
        
        assert len(results) == 3
        for col_name, stats in results.items():
            assert isinstance(stats, StatisticalSummary)
            assert stats.min_value is not None
            assert stats.max_value is not None
    
    def test_non_numeric_data_handling(self):
        """Test handling of non-numeric data."""
        df = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'date_col': pd.date_range('2023-01-01', periods=3),
            'bool_col': [True, False, True]
        })
        
        # Should return empty results for non-numeric columns
        results = self.service.analyze(df)
        assert len(results) == 0
        
        # Distribution analysis should return empty for non-numeric
        text_result = self.service.analyze_distribution(df['text_col'])
        assert text_result == {}
    
    def test_single_value_column(self):
        """Test handling of column with single unique value."""
        df = pd.DataFrame({
            'constant_col': [5, 5, 5, 5, 5]
        })
        
        results = self.service.analyze(df)
        stats = results['constant_col']
        
        assert stats.min_value == 5.0
        assert stats.max_value == 5.0
        assert stats.mean == 5.0
        assert stats.median == 5.0
        assert stats.std_dev == 0.0
    
    @pytest.mark.parametrize("data,expected_skew_sign", [
        ([1, 1, 1, 1, 2, 3, 4, 5], 1),  # Right skewed (positive)
        ([1, 2, 3, 4, 5, 5, 5, 5], -1),  # Left skewed (negative)
    ])
    def test_skewness_detection(self, data, expected_skew_sign):
        """Test skewness detection for different distributions."""
        series = pd.Series(data)
        result = self.service.analyze_distribution(series)
        
        assert 'skewness' in result
        # Check if skewness has the expected sign
        assert np.sign(result['skewness']) == expected_skew_sign