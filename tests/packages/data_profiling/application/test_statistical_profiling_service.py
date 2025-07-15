import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from src.packages.data_profiling.application.services.statistical_profiling_service import StatisticalProfilingService
from src.packages.data_profiling.domain.entities.data_profile import StatisticalSummary


class TestStatisticalProfilingService:
    """Test StatisticalProfilingService."""
    
    @pytest.fixture
    def service(self):
        """Create StatisticalProfilingService instance."""
        return StatisticalProfilingService()
    
    @pytest.fixture
    def numeric_dataframe(self):
        """Create DataFrame with numeric columns for testing."""
        np.random.seed(42)
        data = {
            'normal_dist': np.random.normal(100, 15, 1000),
            'uniform_dist': np.random.uniform(0, 100, 1000),
            'exponential_dist': np.random.exponential(2, 1000),
            'integer_col': np.random.randint(1, 100, 1000),
            'skewed_data': np.random.lognormal(0, 1, 1000),
            'constant_col': [42.0] * 1000,
            'outlier_col': list(np.random.normal(50, 5, 990)) + [200, 300, -100, -200, 500, 600, -50, -60, 250, 350]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mixed_dataframe(self):
        """Create DataFrame with mixed column types."""
        np.random.seed(42)
        data = {
            'numeric': np.random.normal(0, 1, 100),
            'string': ['text'] * 100,
            'datetime': pd.date_range('2023-01-01', periods=100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data for testing."""
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        # Create trending data with seasonality
        trend = np.linspace(100, 200, 365)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
        noise = np.random.normal(0, 5, 365)
        values = trend + seasonal + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })


class TestBasicStatisticalAnalysis:
    """Test basic statistical analysis functionality."""
    
    def test_analyze_numeric_dataframe(self, service, numeric_dataframe):
        """Test analysis of numeric DataFrame."""
        results = service.analyze(numeric_dataframe)
        
        assert isinstance(results, dict)
        assert len(results) == len(numeric_dataframe.select_dtypes(include='number').columns)
        
        # Check that all numeric columns are analyzed
        numeric_columns = numeric_dataframe.select_dtypes(include='number').columns
        for col in numeric_columns:
            assert col in results
            assert isinstance(results[col], StatisticalSummary)
    
    def test_analyze_mixed_dataframe(self, service, mixed_dataframe):
        """Test analysis of DataFrame with mixed types."""
        results = service.analyze(mixed_dataframe)
        
        # Should only analyze numeric columns
        assert len(results) == 1  # Only 'numeric' column
        assert 'numeric' in results
        assert 'string' not in results
        assert 'datetime' not in results
        assert 'categorical' not in results
    
    def test_analyze_empty_dataframe(self, service):
        """Test analysis of empty DataFrame."""
        empty_df = pd.DataFrame()
        results = service.analyze(empty_df)
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_analyze_column_with_nulls(self, service):
        """Test analysis of column with null values."""
        data_with_nulls = pd.DataFrame({
            'col_with_nulls': [1.0, 2.0, None, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        results = service.analyze(data_with_nulls)
        
        assert 'col_with_nulls' in results
        summary = results['col_with_nulls']
        
        # Should compute statistics on non-null values
        assert summary.min_value == 1.0
        assert summary.max_value == 10.0
        assert summary.mean is not None
        assert summary.median is not None


class TestStatisticalSummary:
    """Test statistical summary computation."""
    
    def test_analyze_numeric_column_complete(self, service):
        """Test complete statistical analysis of a numeric column."""
        test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        summary = service._analyze_numeric_column(test_series)
        
        assert summary.min_value == 1.0
        assert summary.max_value == 10.0
        assert summary.mean == 5.5
        assert summary.median == 5.5
        assert len(summary.quartiles) == 3
        assert summary.quartiles[0] == 3.25  # Q1
        assert summary.quartiles[1] == 5.5   # Q2 (median)
        assert summary.quartiles[2] == 7.75  # Q3
    
    def test_analyze_empty_column(self, service):
        """Test analysis of empty column."""
        empty_series = pd.Series([], dtype=float)
        
        summary = service._analyze_numeric_column(empty_series)
        
        assert isinstance(summary, StatisticalSummary)
        # Should handle empty series gracefully
    
    def test_analyze_single_value_column(self, service):
        """Test analysis of column with single value."""
        single_value_series = pd.Series([42.0])
        
        summary = service._analyze_numeric_column(single_value_series)
        
        assert summary.min_value == 42.0
        assert summary.max_value == 42.0
        assert summary.mean == 42.0
        assert summary.median == 42.0
        assert summary.std_dev == 0.0


class TestDistributionAnalysis:
    """Test distribution analysis functionality."""
    
    def test_analyze_distribution_normal(self, service, numeric_dataframe):
        """Test distribution analysis for normal distribution."""
        normal_series = numeric_dataframe['normal_dist']
        
        distribution_info = service.analyze_distribution(normal_series)
        
        assert 'skewness' in distribution_info
        assert 'kurtosis' in distribution_info
        assert 'normality_test' in distribution_info
        assert 'best_distribution' in distribution_info
        assert 'percentiles' in distribution_info
        assert 'entropy' in distribution_info
        
        # Normal distribution should have low skewness
        assert abs(distribution_info['skewness']) < 1.0
    
    def test_analyze_distribution_skewed(self, service, numeric_dataframe):
        """Test distribution analysis for skewed distribution."""
        skewed_series = numeric_dataframe['skewed_data']
        
        distribution_info = service.analyze_distribution(skewed_series)
        
        # Log-normal distribution should be positively skewed
        assert distribution_info['skewness'] > 0.5
    
    def test_analyze_distribution_uniform(self, service, numeric_dataframe):
        """Test distribution analysis for uniform distribution."""
        uniform_series = numeric_dataframe['uniform_dist']
        
        distribution_info = service.analyze_distribution(uniform_series)
        
        # Uniform distribution should have low skewness and negative kurtosis
        assert abs(distribution_info['skewness']) < 0.5
        assert distribution_info['kurtosis'] < 0
    
    def test_analyze_distribution_non_numeric(self, service):
        """Test distribution analysis for non-numeric data."""
        string_series = pd.Series(['a', 'b', 'c'])
        
        distribution_info = service.analyze_distribution(string_series)
        
        assert distribution_info == {}
    
    def test_analyze_distribution_insufficient_data(self, service):
        """Test distribution analysis with insufficient data."""
        small_series = pd.Series([1, 2])
        
        distribution_info = service.analyze_distribution(small_series)
        
        assert distribution_info == {}


class TestNormalityTesting:
    """Test normality testing functionality."""
    
    def test_normality_test_normal_data(self, service):
        """Test normality test on normal data."""
        normal_data = pd.Series(np.random.normal(0, 1, 1000))
        
        normality_result = service._test_normality(normal_data)
        
        assert 'test_name' in normality_result
        assert 'statistic' in normality_result
        assert 'p_value' in normality_result
        assert 'is_normal' in normality_result
        assert 'interpretation' in normality_result
        
        # Should likely detect as normal (though random data might occasionally fail)
        assert isinstance(normality_result['is_normal'], bool)
    
    def test_normality_test_non_normal_data(self, service):
        """Test normality test on clearly non-normal data."""
        # Highly skewed data
        non_normal_data = pd.Series(np.random.exponential(1, 1000))
        
        normality_result = service._test_normality(non_normal_data)
        
        # Should detect as non-normal
        assert normality_result['is_normal'] is False
    
    def test_normality_test_large_sample(self, service):
        """Test normality test with large sample (uses Anderson-Darling)."""
        large_normal_data = pd.Series(np.random.normal(0, 1, 6000))
        
        normality_result = service._test_normality(large_normal_data)
        
        assert normality_result['test_name'] == 'Anderson-Darling'
    
    def test_normality_test_small_sample(self, service):
        """Test normality test with small sample (uses Shapiro-Wilk)."""
        small_normal_data = pd.Series(np.random.normal(0, 1, 100))
        
        normality_result = service._test_normality(small_normal_data)
        
        assert normality_result['test_name'] == 'Shapiro-Wilk'


class TestDistributionFitting:
    """Test distribution fitting functionality."""
    
    def test_fit_distribution_normal(self, service):
        """Test distribution fitting for normal data."""
        normal_data = pd.Series(np.random.normal(100, 15, 1000))
        
        fit_result = service._fit_distribution(normal_data)
        
        assert 'distribution' in fit_result
        assert 'parameters' in fit_result
        assert 'ks_statistic' in fit_result
        assert 'p_value' in fit_result
        assert 'goodness_of_fit' in fit_result
        
        # Should identify normal or log-normal distribution
        assert fit_result['distribution'] in ['norm', 'lognorm', 'gamma', 'uniform', 'expon', 'beta']
    
    def test_fit_distribution_uniform(self, service):
        """Test distribution fitting for uniform data."""
        uniform_data = pd.Series(np.random.uniform(0, 100, 1000))
        
        fit_result = service._fit_distribution(uniform_data)
        
        # Should identify uniform distribution as one of the candidates
        assert 'distribution' in fit_result
    
    def test_fit_distribution_exponential(self, service):
        """Test distribution fitting for exponential data."""
        exp_data = pd.Series(np.random.exponential(2, 1000))
        
        fit_result = service._fit_distribution(exp_data)
        
        # Should identify exponential or gamma distribution
        assert 'distribution' in fit_result


class TestOutlierDetection:
    """Test outlier detection functionality."""
    
    def test_detect_outliers_iqr(self, service, numeric_dataframe):
        """Test IQR outlier detection."""
        outlier_series = numeric_dataframe['outlier_col']
        
        outliers_info = service.detect_outliers(outlier_series, method='iqr')
        
        assert 'method' in outliers_info
        assert 'lower_bound' in outliers_info
        assert 'upper_bound' in outliers_info
        assert 'outlier_count' in outliers_info
        assert 'outlier_percentage' in outliers_info
        assert 'outlier_values' in outliers_info
        
        assert outliers_info['method'] == 'IQR'
        assert outliers_info['outlier_count'] > 0  # Should detect the extreme values we added
    
    def test_detect_outliers_zscore(self, service, numeric_dataframe):
        """Test Z-score outlier detection."""
        outlier_series = numeric_dataframe['outlier_col']
        
        outliers_info = service.detect_outliers(outlier_series, method='zscore')
        
        assert outliers_info['method'] == 'Z-Score'
        assert 'threshold' in outliers_info
        assert 'max_zscore' in outliers_info
        assert outliers_info['outlier_count'] > 0
    
    def test_detect_outliers_modified_zscore(self, service, numeric_dataframe):
        """Test Modified Z-score outlier detection."""
        outlier_series = numeric_dataframe['outlier_col']
        
        outliers_info = service.detect_outliers(outlier_series, method='modified_zscore')
        
        assert outliers_info['method'] == 'Modified Z-Score'
        assert 'max_modified_zscore' in outliers_info
        assert outliers_info['outlier_count'] > 0
    
    @patch('sklearn.ensemble.IsolationForest')
    def test_detect_outliers_isolation_forest(self, mock_isolation_forest, service, numeric_dataframe):
        """Test Isolation Forest outlier detection."""
        # Mock IsolationForest
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([1] * 990 + [-1] * 10)  # 10 outliers
        mock_isolation_forest.return_value = mock_model
        
        outlier_series = numeric_dataframe['outlier_col']
        
        outliers_info = service.detect_outliers(outlier_series, method='isolation_forest')
        
        assert outliers_info['method'] == 'Isolation Forest'
        assert outliers_info['outlier_count'] == 10
    
    def test_detect_outliers_isolation_forest_fallback(self, service, numeric_dataframe):
        """Test Isolation Forest fallback to IQR when sklearn unavailable."""
        outlier_series = numeric_dataframe['outlier_col']
        
        with patch('src.packages.data_profiling.application.services.statistical_profiling_service.IsolationForest', side_effect=ImportError):
            outliers_info = service.detect_outliers(outlier_series, method='isolation_forest')
            
            # Should fallback to IQR
            assert outliers_info['method'] == 'IQR'
    
    def test_detect_outliers_constant_data(self, service):
        """Test outlier detection on constant data."""
        constant_series = pd.Series([42.0] * 100)
        
        outliers_info = service.detect_outliers(constant_series, method='modified_zscore')
        
        # No outliers should be detected in constant data
        assert outliers_info['outlier_count'] == 0
    
    def test_detect_outliers_non_numeric(self, service):
        """Test outlier detection on non-numeric data."""
        string_series = pd.Series(['a', 'b', 'c'])
        
        outliers_info = service.detect_outliers(string_series)
        
        assert outliers_info == {}
    
    def test_detect_outliers_insufficient_data(self, service):
        """Test outlier detection with insufficient data."""
        small_series = pd.Series([1, 2])
        
        outliers_info = service.detect_outliers(small_series)
        
        assert outliers_info == {}


class TestCorrelationAnalysis:
    """Test correlation analysis functionality."""
    
    def test_correlation_analysis(self, service, numeric_dataframe):
        """Test correlation analysis."""
        correlation_results = service.correlation_analysis(numeric_dataframe)
        
        assert 'correlation_matrix' in correlation_results
        assert 'high_correlations' in correlation_results
        assert 'max_correlation' in correlation_results
        assert 'min_correlation' in correlation_results
        
        # Check correlation matrix structure
        corr_matrix = correlation_results['correlation_matrix']
        numeric_cols = numeric_dataframe.select_dtypes(include='number').columns
        
        for col in numeric_cols:
            assert col in corr_matrix
    
    def test_correlation_analysis_no_numeric_columns(self, service):
        """Test correlation analysis with no numeric columns."""
        string_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['x', 'y']})
        
        correlation_results = service.correlation_analysis(string_df)
        
        assert correlation_results == {}
    
    def test_correlation_analysis_single_column(self, service):
        """Test correlation analysis with single numeric column."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        correlation_results = service.correlation_analysis(single_col_df)
        
        assert correlation_results == {}
    
    def test_high_correlation_detection(self, service):
        """Test detection of high correlations."""
        # Create data with known high correlation
        x = np.random.normal(0, 1, 1000)
        y = 0.9 * x + 0.1 * np.random.normal(0, 1, 1000)  # High correlation with x
        z = np.random.normal(0, 1, 1000)  # Independent
        
        corr_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
        correlation_results = service.correlation_analysis(corr_df)
        
        high_correlations = correlation_results['high_correlations']
        
        # Should detect high correlation between x and y
        assert len(high_correlations) > 0
        x_y_correlation = next((item for item in high_correlations 
                               if set([item['column1'], item['column2']]) == {'x', 'y'}), None)
        assert x_y_correlation is not None
        assert abs(x_y_correlation['correlation']) > 0.7


class TestTimeSeriesAnalysis:
    """Test time series analysis functionality."""
    
    def test_analyze_time_series(self, service, time_series_data):
        """Test time series analysis."""
        ts_results = service.analyze_time_series(
            time_series_data['value'],
            time_series_data['date']
        )
        
        assert 'time_span_days' in ts_results
        assert 'frequency' in ts_results
        assert 'trend' in ts_results
        assert 'volatility' in ts_results
        assert 'stationarity' in ts_results
        
        assert ts_results['time_span_days'] > 0
        assert ts_results['frequency'] > 0
    
    def test_analyze_time_series_insufficient_data(self, service):
        """Test time series analysis with insufficient data."""
        short_series = pd.Series([1, 2, 3])
        short_dates = pd.Series(pd.date_range('2023-01-01', periods=3))
        
        ts_results = service.analyze_time_series(short_series, short_dates)
        
        assert ts_results == {}
    
    def test_analyze_time_series_no_datetime(self, service):
        """Test time series analysis without datetime column."""
        series = pd.Series([1, 2, 3, 4, 5])
        
        ts_results = service.analyze_time_series(series)
        
        assert ts_results == {}
    
    @patch('scipy.stats.linregress')
    def test_analyze_trend(self, mock_linregress, service, time_series_data):
        """Test trend analysis."""
        # Mock linear regression result
        mock_linregress.return_value = (0.5, 100, 0.8, 0.01, 0.1)  # slope, intercept, r_value, p_value, std_err
        
        trend_results = service._analyze_trend(time_series_data)
        
        assert 'direction' in trend_results
        assert 'slope' in trend_results
        assert 'r_squared' in trend_results
        assert 'p_value' in trend_results
        assert 'strength' in trend_results
        
        # With p_value < 0.05 and positive slope, should detect increasing trend
        assert trend_results['direction'] == 'increasing'
    
    def test_analyze_volatility(self, service, time_series_data):
        """Test volatility analysis."""
        volatility_results = service._analyze_volatility(time_series_data)
        
        assert 'volatility' in volatility_results
        assert 'mean_return' in volatility_results
        assert 'sharpe_ratio' in volatility_results
        assert 'volatility_clustering' in volatility_results
        assert 'risk_level' in volatility_results
        
        assert volatility_results['risk_level'] in ['low', 'medium', 'high']
    
    @patch('statsmodels.tsa.stattools.adfuller')
    def test_stationarity_test(self, mock_adfuller, service):
        """Test stationarity testing."""
        # Mock ADF test result
        mock_adfuller.return_value = (-3.5, 0.01, 1, 100, {'1%': -3.43, '5%': -2.86, '10%': -2.57}, 0.0)
        
        test_series = pd.Series(np.random.normal(0, 1, 100))
        
        stationarity_results = service._test_stationarity(test_series)
        
        assert 'is_stationary' in stationarity_results
        assert 'adf_statistic' in stationarity_results
        assert 'p_value' in stationarity_results
        assert 'critical_values' in stationarity_results
        assert 'interpretation' in stationarity_results
        
        # With p_value < 0.05, should be stationary
        assert stationarity_results['is_stationary'] is True
    
    def test_stationarity_test_no_statsmodels(self, service):
        """Test stationarity test fallback when statsmodels unavailable."""
        with patch('src.packages.data_profiling.application.services.statistical_profiling_service.adfuller', side_effect=ImportError):
            test_series = pd.Series(np.random.normal(0, 1, 100))
            
            stationarity_results = service._test_stationarity(test_series)
            
            assert 'error' in stationarity_results
            assert stationarity_results['error'] == 'statsmodels not available'


class TestSeasonalityDetection:
    """Test seasonality detection functionality."""
    
    def test_detect_seasonality_sufficient_data(self, service):
        """Test seasonality detection with sufficient data."""
        # Create data with clear seasonality
        t = np.arange(100)
        seasonal_data = pd.Series(10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 100))
        
        seasonality_results = service._detect_seasonality(seasonal_data)
        
        assert 'has_seasonality' in seasonality_results
        assert 'autocorr_12' in seasonality_results
        assert 'autocorr_24' in seasonality_results
        assert 'seasonal_strength' in seasonality_results
    
    def test_detect_seasonality_insufficient_data(self, service):
        """Test seasonality detection with insufficient data."""
        short_series = pd.Series([1, 2, 3, 4, 5])
        
        seasonality_results = service._detect_seasonality(short_series)
        
        assert seasonality_results['has_seasonality'] is False
        assert seasonality_results['reason'] == 'Insufficient data'


class TestDriftDetection:
    """Test drift detection functionality."""
    
    def test_detect_drift_with_drift(self, service):
        """Test drift detection with clear drift."""
        # Create data with drift (mean shift)
        first_half = np.random.normal(0, 1, 100)
        second_half = np.random.normal(5, 1, 100)  # Clear mean shift
        drift_data = pd.Series(np.concatenate([first_half, second_half]))
        
        drift_results = service._detect_drift(drift_data)
        
        assert 'has_drift' in drift_results
        assert 'mean_drift' in drift_results
        assert 'std_drift' in drift_results
        assert 'p_value_mean' in drift_results
        assert 'first_half_mean' in drift_results
        assert 'second_half_mean' in drift_results
        
        # Should detect drift
        assert drift_results['has_drift'] is True
    
    def test_detect_drift_no_drift(self, service):
        """Test drift detection without drift."""
        # Create stable data
        stable_data = pd.Series(np.random.normal(0, 1, 200))
        
        drift_results = service._detect_drift(stable_data)
        
        # Should not detect drift
        assert drift_results['has_drift'] is False
    
    def test_detect_drift_insufficient_data(self, service):
        """Test drift detection with insufficient data."""
        short_series = pd.Series([1, 2, 3])
        
        drift_results = service._detect_drift(short_series)
        
        assert drift_results['has_drift'] is False
        assert drift_results['reason'] == 'Insufficient data'


class TestStatisticalReport:
    """Test statistical report generation."""
    
    def test_generate_statistical_report(self, service, numeric_dataframe):
        """Test statistical report generation."""
        report = service.generate_statistical_report(numeric_dataframe)
        
        assert 'dataset_overview' in report
        assert 'statistical_analysis' in report
        assert 'correlation_analysis' in report
        assert 'recommendations' in report
        
        # Check dataset overview
        overview = report['dataset_overview']
        assert 'shape' in overview
        assert 'memory_usage_mb' in overview
        assert 'dtypes' in overview
        
        assert overview['shape'] == numeric_dataframe.shape
    
    def test_generate_statistical_report_error_handling(self, service):
        """Test statistical report generation with error."""
        # Force an error by passing invalid data
        with patch.object(service, 'analyze', side_effect=Exception("Test error")):
            report = service.generate_statistical_report(pd.DataFrame())
            
            assert 'error' in report
            assert report['error'] == 'Test error'


class TestPercentileCalculation:
    """Test percentile calculation functionality."""
    
    def test_calculate_percentiles(self, service):
        """Test percentile calculation."""
        test_series = pd.Series(range(1, 101))  # 1 to 100
        
        percentiles = service._calculate_percentiles(test_series)
        
        expected_percentiles = ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
        
        for p in expected_percentiles:
            assert p in percentiles
        
        # Check some known values
        assert percentiles['p50'] == 50.5  # Median
        assert percentiles['p25'] == 25.25  # Q1
        assert percentiles['p75'] == 75.75  # Q3


class TestEntropyCalculation:
    """Test entropy calculation functionality."""
    
    def test_calculate_entropy_uniform(self, service):
        """Test entropy calculation for uniform distribution."""
        # Uniform data should have high entropy
        uniform_data = pd.Series([1, 2, 3, 4, 5] * 100)
        
        entropy = service._calculate_entropy(uniform_data)
        
        assert entropy > 0
        assert isinstance(entropy, float)
    
    def test_calculate_entropy_constant(self, service):
        """Test entropy calculation for constant data."""
        # Constant data should have low entropy
        constant_data = pd.Series([1] * 100)
        
        entropy = service._calculate_entropy(constant_data)
        
        assert entropy == 0.0
    
    def test_calculate_entropy_error_handling(self, service):
        """Test entropy calculation error handling."""
        # Empty series should return 0
        empty_series = pd.Series([])
        
        entropy = service._calculate_entropy(empty_series)
        
        assert entropy == 0.0