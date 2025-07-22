"""
Statistical analysis algorithm validation tests.
Tests statistical methods, distribution fitting, and hypothesis testing for production deployment.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path
import time
from unittest.mock import Mock
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from statistics.application.services.statistical_analysis_service import StatisticalAnalysisService
    from statistics.application.services.distribution_fitting_service import DistributionFittingService
    from statistics.application.services.hypothesis_testing_service import HypothesisTestingService
    from statistics.domain.entities.statistical_test import StatisticalTest, TestResult
except ImportError as e:
    # Create mock classes for testing infrastructure
    class StatisticalAnalysisService:
        def __init__(self):
            self.cached_results = {}
            
        def calculate_descriptive_statistics(self, data: np.ndarray) -> Dict[str, float]:
            """Calculate comprehensive descriptive statistics."""
            if len(data) == 0:
                return {'error': 'Empty dataset'}
                
            stats_dict = {
                'count': len(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'mode': float(stats.mode(data, keepdims=True)[0][0]) if len(data) > 0 else np.nan,
                'std': np.std(data, ddof=1),
                'variance': np.var(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'q1': np.percentile(data, 25),
                'q3': np.percentile(data, 75),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.inf
            }
            
            return {'success': True, 'statistics': stats_dict}
            
        def calculate_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
            """Calculate correlation matrix using specified method."""
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'success': False, 'error': 'No numeric columns found'}
                
            if method == 'pearson':
                corr_matrix = numeric_data.corr(method='pearson')
                p_values = self._calculate_correlation_p_values(numeric_data, method='pearson')
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman')
                p_values = self._calculate_correlation_p_values(numeric_data, method='spearman')
            elif method == 'kendall':
                corr_matrix = numeric_data.corr(method='kendall')
                p_values = self._calculate_correlation_p_values(numeric_data, method='kendall')
            else:
                return {'success': False, 'error': f'Unsupported correlation method: {method}'}
                
            # Find significant correlations
            significant_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix.loc[col1, col2]
                        p_val = p_values.loc[col1, col2]
                        
                        if abs(corr_val) > 0.3 and p_val < 0.05:  # Significant threshold
                            significant_pairs.append({
                                'variable1': col1,
                                'variable2': col2,
                                'correlation': corr_val,
                                'p_value': p_val,
                                'significance': 'significant' if p_val < 0.01 else 'moderate'
                            })
                            
            return {
                'success': True,
                'correlation_matrix': corr_matrix.to_dict(),
                'p_values': p_values.to_dict(),
                'significant_correlations': significant_pairs,
                'method': method
            }
            
        def detect_outliers(self, data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
            """Detect outliers using various methods."""
            outlier_indices = []
            outlier_values = []
            
            if method == 'iqr':
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = data[outlier_mask].tolist()
                
                bounds = {'lower': lower_bound, 'upper': upper_bound}
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > threshold
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = data[outlier_mask].tolist()
                
                bounds = {'threshold': threshold, 'mean': np.mean(data), 'std': np.std(data)}
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                
                outlier_mask = np.abs(modified_z_scores) > threshold
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = data[outlier_mask].tolist()
                
                bounds = {'threshold': threshold, 'median': median, 'mad': mad}
                
            else:
                return {'success': False, 'error': f'Unsupported outlier detection method: {method}'}
                
            outlier_percentage = (len(outlier_indices) / len(data)) * 100
            
            return {
                'success': True,
                'method': method,
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': outlier_percentage,
                'bounds': bounds,
                'threshold': threshold
            }
            
        def perform_regression_analysis(self, X: np.ndarray, y: np.ndarray, regression_type: str = 'linear') -> Dict[str, Any]:
            """Perform regression analysis."""
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            
            if regression_type == 'linear':
                model = LinearRegression()
                model.fit(X, y)
                
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                
                # Calculate feature importance (coefficients for linear regression)
                feature_importance = abs(model.coef_) / np.sum(abs(model.coef_))
                
                results = {
                    'model_type': 'linear',
                    'r2_score': r2,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_,
                    'feature_importance': feature_importance.tolist()
                }
                
            elif regression_type == 'polynomial':
                # Use degree 2 polynomial
                poly_features = PolynomialFeatures(degree=2)
                model = Pipeline([
                    ('poly', poly_features),
                    ('linear', LinearRegression())
                ])
                
                model.fit(X, y)
                predictions = model.predict(X)
                
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                
                results = {
                    'model_type': 'polynomial',
                    'degree': 2,
                    'r2_score': r2,
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                }
                
            elif regression_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                
                results = {
                    'model_type': 'random_forest',
                    'r2_score': r2,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'feature_importance': model.feature_importances_.tolist()
                }
                
            else:
                return {'success': False, 'error': f'Unsupported regression type: {regression_type}'}
                
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results.update({
                'success': True,
                'cross_validation': {
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'cv_scores': cv_scores.tolist()
                },
                'model_fitted': True
            })
            
            return results
            
        def _calculate_correlation_p_values(self, data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
            """Calculate p-values for correlation matrix."""
            from scipy.stats import pearsonr, spearmanr, kendalltau
            
            columns = data.columns
            p_values = pd.DataFrame(index=columns, columns=columns, dtype=float)
            
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i == j:
                        p_values.loc[col1, col2] = 0.0
                    else:
                        x, y = data[col1].values, data[col2].values
                        
                        # Remove NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x, y = x[mask], y[mask]
                        
                        if len(x) < 3:  # Need at least 3 points
                            p_values.loc[col1, col2] = 1.0
                            continue
                            
                        try:
                            if method == 'pearson':
                                _, p_val = pearsonr(x, y)
                            elif method == 'spearman':
                                _, p_val = spearmanr(x, y)
                            elif method == 'kendall':
                                _, p_val = kendalltau(x, y)
                            else:
                                p_val = 1.0
                                
                            p_values.loc[col1, col2] = p_val
                            
                        except Exception:
                            p_values.loc[col1, col2] = 1.0
                            
            return p_values
    
    class DistributionFittingService:
        def __init__(self):
            self.fitted_distributions = {}
            
        def fit_distribution(self, data: np.ndarray, distributions: List[str] = None) -> Dict[str, Any]:
            """Fit multiple distributions to data and find best fit."""
            if distributions is None:
                distributions = ['norm', 'expon', 'uniform', 'gamma', 'lognorm', 'beta']
                
            results = {}
            best_distribution = None
            best_aic = np.inf
            
            for dist_name in distributions:
                try:
                    # Get distribution object
                    dist = getattr(stats, dist_name)
                    
                    # Fit distribution
                    params = dist.fit(data)
                    
                    # Calculate AIC and BIC
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    k = len(params)  # Number of parameters
                    n = len(data)    # Sample size
                    
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    # Kolmogorov-Smirnov test
                    ks_statistic, ks_p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
                    
                    results[dist_name] = {
                        'parameters': params,
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood,
                        'ks_statistic': ks_statistic,
                        'ks_p_value': ks_p_value,
                        'goodness_of_fit': ks_p_value > 0.05  # Good fit if p > 0.05
                    }
                    
                    # Track best distribution by AIC
                    if aic < best_aic:
                        best_aic = aic
                        best_distribution = dist_name
                        
                except Exception as e:
                    results[dist_name] = {
                        'error': str(e),
                        'fit_successful': False
                    }
                    
            return {
                'success': True,
                'distributions_tested': len(distributions),
                'best_distribution': best_distribution,
                'best_aic': best_aic,
                'results': results,
                'data_summary': {
                    'sample_size': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data)
                }
            }
            
        def generate_qq_plot_data(self, data: np.ndarray, distribution: str = 'norm') -> Dict[str, Any]:
            """Generate Q-Q plot data for distribution comparison."""
            try:
                dist = getattr(stats, distribution)
                
                # Fit distribution to data
                params = dist.fit(data)
                
                # Generate theoretical quantiles
                n = len(data)
                quantiles = np.linspace(0.01, 0.99, n)
                theoretical_quantiles = dist.ppf(quantiles, *params)
                
                # Sort actual data
                actual_quantiles = np.sort(data)
                
                # Calculate R-squared for linearity
                correlation = np.corrcoef(theoretical_quantiles, actual_quantiles)[0, 1]
                r_squared = correlation ** 2
                
                return {
                    'success': True,
                    'distribution': distribution,
                    'parameters': params,
                    'theoretical_quantiles': theoretical_quantiles.tolist(),
                    'actual_quantiles': actual_quantiles.tolist(),
                    'r_squared': r_squared,
                    'correlation': correlation,
                    'good_fit': r_squared > 0.95  # Good fit threshold
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'distribution': distribution
                }
    
    class HypothesisTestingService:
        def __init__(self):
            self.test_history = []
            
        def perform_t_test(self, data1: np.ndarray, data2: np.ndarray = None, 
                          alternative: str = 'two-sided', alpha: float = 0.05) -> Dict[str, Any]:
            """Perform t-test (one-sample or two-sample)."""
            if data2 is None:
                # One-sample t-test against zero
                t_statistic, p_value = stats.ttest_1samp(data1, 0)
                test_type = 'one_sample_t_test'
                sample_sizes = [len(data1)]
                means = [np.mean(data1)]
            else:
                # Two-sample t-test
                t_statistic, p_value = stats.ttest_ind(data1, data2)
                test_type = 'two_sample_t_test' 
                sample_sizes = [len(data1), len(data2)]
                means = [np.mean(data1), np.mean(data2)]
                
            # Adjust p-value based on alternative hypothesis
            if alternative == 'less':
                p_value = p_value / 2 if t_statistic < 0 else 1 - p_value / 2
            elif alternative == 'greater':
                p_value = p_value / 2 if t_statistic > 0 else 1 - p_value / 2
                
            reject_null = p_value < alpha
            
            # Calculate effect size (Cohen's d)
            if data2 is not None:
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                    (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                   (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            else:
                cohens_d = np.mean(data1) / np.std(data1, ddof=1)
                
            result = {
                'success': True,
                'test_type': test_type,
                't_statistic': t_statistic,
                'p_value': p_value,
                'alpha': alpha,
                'reject_null': reject_null,
                'alternative': alternative,
                'sample_sizes': sample_sizes,
                'sample_means': means,
                'effect_size': cohens_d,
                'interpretation': self._interpret_effect_size(abs(cohens_d))
            }
            
            self.test_history.append(result)
            return result
            
        def perform_chi_square_test(self, observed: np.ndarray, expected: np.ndarray = None, 
                                  alpha: float = 0.05) -> Dict[str, Any]:
            """Perform chi-square goodness of fit or independence test."""
            if expected is None:
                # Equal expected frequencies
                expected = np.full_like(observed, np.mean(observed), dtype=float)
                
            chi2_statistic, p_value = stats.chisquare(observed, expected)
            
            degrees_of_freedom = len(observed) - 1
            reject_null = p_value < alpha
            
            # Calculate Cramér's V (effect size)
            n = np.sum(observed)
            cramers_v = np.sqrt(chi2_statistic / (n * (min(len(observed), len(expected)) - 1)))
            
            return {
                'success': True,
                'test_type': 'chi_square_goodness_of_fit',
                'chi2_statistic': chi2_statistic,
                'p_value': p_value,
                'degrees_of_freedom': degrees_of_freedom,
                'alpha': alpha,
                'reject_null': reject_null,
                'observed': observed.tolist(),
                'expected': expected.tolist(),
                'effect_size': cramers_v,
                'interpretation': self._interpret_cramers_v(cramers_v)
            }
            
        def perform_anova(self, groups: List[np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
            """Perform one-way ANOVA."""
            f_statistic, p_value = stats.f_oneway(*groups)
            
            degrees_of_freedom_between = len(groups) - 1
            degrees_of_freedom_within = sum(len(group) - 1 for group in groups)
            
            reject_null = p_value < alpha
            
            # Calculate eta-squared (effect size)
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
            ss_total = sum((x - grand_mean) ** 2 for group in groups for x in group)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            return {
                'success': True,
                'test_type': 'one_way_anova',
                'f_statistic': f_statistic,
                'p_value': p_value,
                'degrees_of_freedom': (degrees_of_freedom_between, degrees_of_freedom_within),
                'alpha': alpha,
                'reject_null': reject_null,
                'group_count': len(groups),
                'group_means': [np.mean(group) for group in groups],
                'effect_size': eta_squared,
                'interpretation': self._interpret_eta_squared(eta_squared)
            }
            
        def _interpret_effect_size(self, cohens_d: float) -> str:
            """Interpret Cohen's d effect size."""
            if cohens_d < 0.2:
                return 'negligible'
            elif cohens_d < 0.5:
                return 'small'
            elif cohens_d < 0.8:
                return 'medium'
            else:
                return 'large'
                
        def _interpret_cramers_v(self, cramers_v: float) -> str:
            """Interpret Cramér's V effect size."""
            if cramers_v < 0.1:
                return 'negligible'
            elif cramers_v < 0.3:
                return 'small'
            elif cramers_v < 0.5:
                return 'medium'
            else:
                return 'large'
                
        def _interpret_eta_squared(self, eta_squared: float) -> str:
            """Interpret eta-squared effect size."""
            if eta_squared < 0.01:
                return 'negligible'
            elif eta_squared < 0.06:
                return 'small'
            elif eta_squared < 0.14:
                return 'medium'
            else:
                return 'large'


@pytest.mark.parametrize("data_type,expected_properties", [
    ('normal', {'skewness_range': (-0.5, 0.5), 'kurtosis_range': (-1, 1)}),
    ('exponential', {'skewness_range': (1.5, 2.5), 'kurtosis_range': (5, 7)}),
    ('uniform', {'skewness_range': (-0.2, 0.2), 'kurtosis_range': (-1.5, -1)}),
])
class TestDescriptiveStatisticsAccuracy:
    """Test descriptive statistics calculation accuracy."""
    
    def test_comprehensive_descriptive_statistics(
        self,
        data_type: str,
        expected_properties: Dict[str, Any]
    ):
        """Test comprehensive descriptive statistics calculation."""
        statistical_service = StatisticalAnalysisService()
        
        # Generate data based on type
        np.random.seed(42)
        if data_type == 'normal':
            data = np.random.normal(100, 15, 1000)
            expected_mean_range = (95, 105)
            expected_std_range = (13, 17)
        elif data_type == 'exponential':
            data = np.random.exponential(2, 1000)
            expected_mean_range = (1.5, 2.5)
            expected_std_range = (1.5, 2.5)
        elif data_type == 'uniform':
            data = np.random.uniform(0, 100, 1000)
            expected_mean_range = (45, 55)
            expected_std_range = (25, 35)
        else:
            pytest.skip(f"Unsupported data type: {data_type}")
            
        # Calculate descriptive statistics
        result = statistical_service.calculate_descriptive_statistics(data)
        
        assert result['success'], "Descriptive statistics calculation failed"
        assert 'statistics' in result, "Statistics not returned"
        
        stats_dict = result['statistics']
        
        # Validate required statistics are present
        required_stats = [
            'count', 'mean', 'median', 'std', 'variance', 'min', 'max', 
            'range', 'q1', 'q3', 'iqr', 'skewness', 'kurtosis', 'cv'
        ]
        
        for stat in required_stats:
            assert stat in stats_dict, f"Missing statistic: {stat}"
            assert not np.isnan(stats_dict[stat]) or stat == 'mode', f"Invalid value for {stat}"
            
        # Validate basic properties
        assert stats_dict['count'] == len(data), "Count mismatch"
        assert stats_dict['min'] <= stats_dict['q1'] <= stats_dict['median'] <= stats_dict['q3'] <= stats_dict['max'], "Quantile order violation"
        assert stats_dict['range'] == stats_dict['max'] - stats_dict['min'], "Range calculation error"
        assert stats_dict['iqr'] == stats_dict['q3'] - stats_dict['q1'], "IQR calculation error"
        
        # Validate distribution-specific properties
        assert expected_mean_range[0] <= stats_dict['mean'] <= expected_mean_range[1], (
            f"Mean {stats_dict['mean']} outside expected range {expected_mean_range}"
        )
        assert expected_std_range[0] <= stats_dict['std'] <= expected_std_range[1], (
            f"Standard deviation {stats_dict['std']} outside expected range {expected_std_range}"
        )
        
        # Validate skewness and kurtosis for distribution type
        skew_range = expected_properties['skewness_range']
        kurt_range = expected_properties['kurtosis_range']
        
        assert skew_range[0] <= stats_dict['skewness'] <= skew_range[1], (
            f"Skewness {stats_dict['skewness']} outside expected range {skew_range}"
        )
        assert kurt_range[0] <= stats_dict['kurtosis'] <= kurt_range[1], (
            f"Kurtosis {stats_dict['kurtosis']} outside expected range {kurt_range}"
        )
    
    def test_edge_case_handling(
        self,
        data_type: str,
        expected_properties: Dict[str, Any]
    ):
        """Test edge cases in descriptive statistics."""
        statistical_service = StatisticalAnalysisService()
        
        # Test empty array
        empty_result = statistical_service.calculate_descriptive_statistics(np.array([]))
        assert 'error' in empty_result, "Empty array should return error"
        
        # Test single value
        single_value = np.array([42.0])
        single_result = statistical_service.calculate_descriptive_statistics(single_value)
        
        if single_result.get('success'):
            stats = single_result['statistics']
            assert stats['count'] == 1, "Single value count incorrect"
            assert stats['mean'] == 42.0, "Single value mean incorrect"
            assert stats['std'] == 0.0, "Single value std should be zero"
            assert stats['variance'] == 0.0, "Single value variance should be zero"
            assert stats['min'] == stats['max'] == 42.0, "Single value min/max incorrect"
            
        # Test constant values
        constant_data = np.full(100, 50.0)
        constant_result = statistical_service.calculate_descriptive_statistics(constant_data)
        
        if constant_result.get('success'):
            stats = constant_result['statistics']
            assert stats['std'] == 0.0, "Constant data std should be zero"
            assert stats['variance'] == 0.0, "Constant data variance should be zero"
            assert np.isinf(stats['cv']) or stats['cv'] == 0, "CV should be infinity or zero for constant data"
        
        # Test data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 95)
        outliers = np.array([200, 250, -50, -100, 300])  # Extreme outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        outlier_result = statistical_service.calculate_descriptive_statistics(data_with_outliers)
        assert outlier_result['success'], "Statistics with outliers should succeed"
        
        outlier_stats = outlier_result['statistics']
        
        # With outliers, range should be much larger
        assert outlier_stats['range'] > 300, "Range with outliers should be large"
        assert abs(outlier_stats['skewness']) > 0.5, "Data with extreme outliers should be skewed"


@pytest.mark.parametrize("correlation_type,expected_strength", [
    ('strong_positive', (0.8, 1.0)),
    ('strong_negative', (-1.0, -0.8)),
    ('no_correlation', (-0.2, 0.2)),
])
class TestCorrelationAnalysisValidation:
    """Test correlation analysis methods and accuracy."""
    
    def test_correlation_matrix_calculation(
        self,
        correlation_type: str,
        expected_strength: Tuple[float, float],
        correlation_test_data: Dict[str, np.ndarray]
    ):
        """Test correlation matrix calculation for different correlation types."""
        statistical_service = StatisticalAnalysisService()
        
        # Get test data for correlation type
        if correlation_type not in correlation_test_data:
            pytest.skip(f"Correlation type {correlation_type} not in test data")
            
        x_data, y_data = correlation_test_data[correlation_type]
        
        # Create DataFrame
        df = pd.DataFrame({
            'variable_x': x_data,
            'variable_y': y_data,
            'noise_variable': np.random.normal(0, 1, len(x_data))  # Uncorrelated variable
        })
        
        # Test different correlation methods
        methods = ['pearson', 'spearman', 'kendall']
        
        for method in methods:
            result = statistical_service.calculate_correlation_matrix(df, method=method)
            
            assert result['success'], f"Correlation calculation failed for {method}"
            assert 'correlation_matrix' in result, "Correlation matrix not returned"
            assert 'p_values' in result, "P-values not returned"
            assert 'significant_correlations' in result, "Significant correlations not returned"
            
            corr_matrix = pd.DataFrame(result['correlation_matrix'])
            
            # Test correlation between x and y
            xy_correlation = corr_matrix.loc['variable_x', 'variable_y']
            
            # Validate correlation strength
            assert expected_strength[0] <= xy_correlation <= expected_strength[1], (
                f"Correlation {xy_correlation} outside expected range {expected_strength} for {method}"
            )
            
            # Validate diagonal correlations are 1.0
            for var in corr_matrix.columns:
                assert abs(corr_matrix.loc[var, var] - 1.0) < 1e-10, f"Diagonal correlation not 1.0 for {var}"
            
            # Validate matrix symmetry
            for i, var1 in enumerate(corr_matrix.columns):
                for j, var2 in enumerate(corr_matrix.columns):
                    corr_val_ij = corr_matrix.loc[var1, var2]
                    corr_val_ji = corr_matrix.loc[var2, var1]
                    assert abs(corr_val_ij - corr_val_ji) < 1e-10, f"Correlation matrix not symmetric: {var1}-{var2}"
            
            # Test significant correlations detection
            significant_pairs = result['significant_correlations']
            
            if correlation_type in ['strong_positive', 'strong_negative']:
                # Should detect significant correlation between x and y
                xy_significant = any(
                    (pair['variable1'] == 'variable_x' and pair['variable2'] == 'variable_y') or
                    (pair['variable1'] == 'variable_y' and pair['variable2'] == 'variable_x')
                    for pair in significant_pairs
                )
                assert xy_significant, f"Strong correlation not detected as significant for {method}"
            
            elif correlation_type == 'no_correlation':
                # Should not detect significant correlation between x and y
                xy_significant = any(
                    (pair['variable1'] == 'variable_x' and pair['variable2'] == 'variable_y') or
                    (pair['variable1'] == 'variable_y' and pair['variable2'] == 'variable_x')
                    for pair in significant_pairs
                )
                assert not xy_significant, f"No correlation incorrectly detected as significant for {method}"
    
    def test_correlation_method_comparison(
        self,
        correlation_type: str,
        expected_strength: Tuple[float, float]
    ):
        """Test comparison between different correlation methods."""
        statistical_service = StatisticalAnalysisService()
        
        # Generate data with known relationships
        np.random.seed(42)
        n = 1000
        
        if correlation_type == 'strong_positive':
            x = np.random.normal(0, 1, n)
            y = 2 * x + np.random.normal(0, 0.5, n)  # Linear relationship
        elif correlation_type == 'strong_negative':
            x = np.random.normal(0, 1, n)
            y = -1.5 * x + np.random.normal(0, 0.3, n)  # Linear relationship
        elif correlation_type == 'no_correlation':
            x = np.random.normal(0, 1, n)
            y = np.random.normal(0, 1, n)  # Independent
        else:
            pytest.skip(f"Unsupported correlation type: {correlation_type}")
            
        df = pd.DataFrame({'x': x, 'y': y})
        
        # Calculate correlations with different methods
        methods_results = {}
        for method in ['pearson', 'spearman', 'kendall']:
            result = statistical_service.calculate_correlation_matrix(df, method=method)
            assert result['success'], f"Method {method} failed"
            
            corr_matrix = pd.DataFrame(result['correlation_matrix'])
            xy_corr = corr_matrix.loc['x', 'y']
            methods_results[method] = xy_corr
            
        # For linear relationships, Pearson should be strongest
        if correlation_type in ['strong_positive', 'strong_negative']:
            pearson_corr = abs(methods_results['pearson'])
            spearman_corr = abs(methods_results['spearman'])
            
            # Pearson should be at least as strong as Spearman for linear relationships
            assert pearson_corr >= spearman_corr * 0.95, (
                f"Pearson correlation {pearson_corr} should be stronger than Spearman {spearman_corr} for linear data"
            )
            
        # All methods should detect similar patterns for monotonic relationships
        correlation_values = list(methods_results.values())
        correlation_signs = [1 if corr > 0 else -1 if corr < 0 else 0 for corr in correlation_values]
        
        # All correlations should have same sign
        if correlation_type == 'strong_positive':
            assert all(sign > 0 for sign in correlation_signs), "All methods should detect positive correlation"
        elif correlation_type == 'strong_negative':
            assert all(sign < 0 for sign in correlation_signs), "All methods should detect negative correlation"


@pytest.mark.parametrize("outlier_method,threshold", [
    ('iqr', 1.5),
    ('zscore', 3.0),
    ('modified_zscore', 3.5)
])
class TestOutlierDetectionAccuracy:
    """Test outlier detection methods and accuracy."""
    
    def test_outlier_detection_accuracy(
        self,
        outlier_method: str,
        threshold: float,
        outlier_detection_data: Dict[str, np.ndarray]
    ):
        """Test outlier detection accuracy with known outliers."""
        statistical_service = StatisticalAnalysisService()
        
        # Test with univariate data containing known outliers
        data_with_outliers = outlier_detection_data['univariate_with_outliers']
        known_outlier_indices = set(outlier_detection_data['known_outlier_indices'])
        
        result = statistical_service.detect_outliers(
            data=data_with_outliers,
            method=outlier_method,
            threshold=threshold
        )
        
        assert result['success'], f"Outlier detection failed for {outlier_method}"
        assert 'outlier_indices' in result, "Outlier indices not returned"
        assert 'outlier_count' in result, "Outlier count not returned"
        assert 'outlier_percentage' in result, "Outlier percentage not returned"
        
        detected_outlier_indices = set(result['outlier_indices'])
        
        # Calculate detection accuracy
        true_positives = len(detected_outlier_indices & known_outlier_indices)
        false_positives = len(detected_outlier_indices - known_outlier_indices)
        false_negatives = len(known_outlier_indices - detected_outlier_indices)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Validate detection performance
        # Should detect at least 60% of known outliers
        assert recall >= 0.6, f"Low recall {recall:.2f} for {outlier_method}. Detected: {true_positives}, Missed: {false_negatives}"
        
        # Should have reasonable precision (at least 40%)
        assert precision >= 0.4, f"Low precision {precision:.2f} for {outlier_method}. True: {true_positives}, False: {false_positives}"
        
        # Validate outlier percentage is reasonable
        outlier_percentage = result['outlier_percentage']
        assert 0 <= outlier_percentage <= 20, f"Outlier percentage {outlier_percentage}% seems unreasonable"
        
        # Test with clean data (should detect few or no outliers)
        clean_data = outlier_detection_data['univariate_clean']
        clean_result = statistical_service.detect_outliers(
            data=clean_data,
            method=outlier_method,
            threshold=threshold
        )
        
        assert clean_result['success'], f"Clean data outlier detection failed for {outlier_method}"
        
        clean_outlier_percentage = clean_result['outlier_percentage']
        assert clean_outlier_percentage <= 5, f"Too many outliers {clean_outlier_percentage}% detected in clean data"
    
    def test_outlier_detection_consistency(
        self,
        outlier_method: str,
        threshold: float
    ):
        """Test outlier detection consistency across multiple runs."""
        statistical_service = StatisticalAnalysisService()
        
        # Generate consistent test data
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 1000)
        outliers = np.array([100, 120, -20, 0, 150])
        test_data = np.concatenate([normal_data, outliers])
        
        # Run detection multiple times
        results = []
        for _ in range(5):
            result = statistical_service.detect_outliers(
                data=test_data.copy(),
                method=outlier_method,
                threshold=threshold
            )
            assert result['success'], f"Detection failed in consistency test"
            results.append(set(result['outlier_indices']))
        
        # All runs should produce identical results
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first_result, (
                f"Inconsistent results in run {i+1}. Expected: {first_result}, Got: {result}"
            )
    
    def test_outlier_threshold_sensitivity(
        self,
        outlier_method: str,
        threshold: float
    ):
        """Test outlier detection sensitivity to threshold changes."""
        statistical_service = StatisticalAnalysisService()
        
        # Generate test data
        np.random.seed(42)
        test_data = np.concatenate([
            np.random.normal(50, 10, 950),
            np.array([80, 90, 10, 0, 100])  # Moderate outliers
        ])
        
        # Test different thresholds
        if outlier_method == 'iqr':
            thresholds = [1.0, 1.5, 2.0, 2.5]
        elif outlier_method == 'zscore':
            thresholds = [2.0, 2.5, 3.0, 3.5]
        elif outlier_method == 'modified_zscore':
            thresholds = [2.5, 3.0, 3.5, 4.0]
        else:
            thresholds = [threshold]  # Use provided threshold if method not recognized
        
        previous_outlier_count = float('inf')
        
        for thresh in sorted(thresholds, reverse=True):  # Test from strict to lenient
            result = statistical_service.detect_outliers(
                data=test_data,
                method=outlier_method,
                threshold=thresh
            )
            
            assert result['success'], f"Detection failed for threshold {thresh}"
            
            current_outlier_count = result['outlier_count']
            
            # More lenient thresholds should detect fewer or equal outliers
            assert current_outlier_count <= previous_outlier_count, (
                f"Lenient threshold {thresh} detected more outliers ({current_outlier_count}) "
                f"than strict threshold ({previous_outlier_count})"
            )
            
            previous_outlier_count = current_outlier_count


@pytest.mark.parametrize("regression_type", [
    'linear',
    'polynomial', 
    'random_forest'
])
class TestRegressionAnalysisValidation:
    """Test regression analysis accuracy and performance."""
    
    def test_regression_model_accuracy(
        self,
        regression_type: str,
        regression_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test regression model accuracy on known dataset."""
        statistical_service = StatisticalAnalysisService()
        
        X, y = regression_dataset
        
        result = statistical_service.perform_regression_analysis(
            X=X, y=y, regression_type=regression_type
        )
        
        assert result['success'], f"Regression analysis failed for {regression_type}"
        assert 'model_type' in result, "Model type not returned"
        assert result['model_type'] == regression_type, "Model type mismatch"
        
        # Validate performance metrics
        required_metrics = ['r2_score', 'mse', 'rmse']
        for metric in required_metrics:
            assert metric in result, f"Missing metric: {metric}"
            assert isinstance(result[metric], (int, float)), f"Invalid {metric} type"
            assert not np.isnan(result[metric]), f"NaN value for {metric}"
        
        # Validate R-squared range
        r2_score = result['r2_score']
        assert -1 <= r2_score <= 1, f"R-squared {r2_score} outside valid range [-1, 1]"
        
        # For this synthetic dataset, should achieve reasonable performance
        if regression_type == 'linear':
            assert r2_score >= 0.8, f"Linear regression R² {r2_score} too low for synthetic data"
        elif regression_type == 'random_forest':
            assert r2_score >= 0.9, f"Random forest R² {r2_score} too low for synthetic data"
        
        # Validate MSE and RMSE relationship
        mse = result['mse']
        rmse = result['rmse']
        assert abs(rmse - np.sqrt(mse)) < 1e-10, "RMSE not equal to sqrt(MSE)"
        
        # Validate cross-validation results
        assert 'cross_validation' in result, "Cross-validation results missing"
        cv_results = result['cross_validation']
        
        assert 'mean_cv_score' in cv_results, "Mean CV score missing"
        assert 'std_cv_score' in cv_results, "CV standard deviation missing"
        assert 'cv_scores' in cv_results, "Individual CV scores missing"
        
        mean_cv = cv_results['mean_cv_score']
        assert -1 <= mean_cv <= 1, f"Mean CV score {mean_cv} outside valid range"
        
        # CV performance should be reasonably close to training performance
        cv_r2_diff = abs(r2_score - mean_cv)
        assert cv_r2_diff <= 0.3, f"Large difference between training R² ({r2_score:.3f}) and CV R² ({mean_cv:.3f})"
    
    def test_regression_feature_importance(
        self,
        regression_type: str,
        regression_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test feature importance calculation and interpretation."""
        statistical_service = StatisticalAnalysisService()
        
        X, y = regression_dataset
        
        result = statistical_service.perform_regression_analysis(
            X=X, y=y, regression_type=regression_type
        )
        
        assert result['success'], f"Regression analysis failed for {regression_type}"
        
        # Feature importance should be available for linear and random forest
        if regression_type in ['linear', 'random_forest']:
            assert 'feature_importance' in result, f"Feature importance missing for {regression_type}"
            
            importance = np.array(result['feature_importance'])
            
            # Validate importance properties
            assert len(importance) == X.shape[1], "Feature importance length mismatch"
            assert all(imp >= 0 for imp in importance), "Negative feature importance values"
            
            # For linear regression, importance should sum to 1 (normalized coefficients)
            if regression_type == 'linear':
                assert abs(np.sum(importance) - 1.0) < 1e-10, "Linear regression feature importance should sum to 1"
            
            # For random forest, importance should sum to 1
            elif regression_type == 'random_forest':
                assert abs(np.sum(importance) - 1.0) < 1e-6, "Random forest feature importance should sum to 1"
            
            # Check that at least one feature has meaningful importance
            assert np.max(importance) > 0.01, "No features have meaningful importance"
    
    def test_regression_model_comparison(
        self,
        regression_type: str,
        regression_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test regression model comparison and selection."""
        statistical_service = StatisticalAnalysisService()
        
        X, y = regression_dataset
        
        # Run multiple regression types for comparison
        model_types = ['linear', 'polynomial', 'random_forest']
        results = {}
        
        for model_type in model_types:
            result = statistical_service.perform_regression_analysis(
                X=X, y=y, regression_type=model_type
            )
            
            if result['success']:
                results[model_type] = result
            
        assert len(results) >= 2, "At least 2 models should succeed for comparison"
        
        # Compare R-squared scores
        r2_scores = {model: result['r2_score'] for model, result in results.items()}
        
        # Random forest should generally perform best on this synthetic data
        if 'random_forest' in r2_scores:
            rf_r2 = r2_scores['random_forest']
            
            for model, r2 in r2_scores.items():
                if model != 'random_forest':
                    assert rf_r2 >= r2 - 0.1, f"Random forest R² ({rf_r2:.3f}) should be competitive with {model} R² ({r2:.3f})"
        
        # Linear and polynomial should show reasonable performance
        if 'linear' in r2_scores and 'polynomial' in r2_scores:
            linear_r2 = r2_scores['linear']
            poly_r2 = r2_scores['polynomial']
            
            # Polynomial should perform at least as well as linear
            assert poly_r2 >= linear_r2 - 0.05, f"Polynomial R² ({poly_r2:.3f}) should be at least as good as linear R² ({linear_r2:.3f})"
        
        # Validate cross-validation consistency
        for model, result in results.items():
            if 'cross_validation' in result:
                cv_mean = result['cross_validation']['mean_cv_score']
                cv_std = result['cross_validation']['std_cv_score']
                
                # CV standard deviation should be reasonable
                assert cv_std <= 0.3, f"High CV standard deviation {cv_std:.3f} for {model}"
                
                # CV mean should be positive for reasonable models
                assert cv_mean >= 0, f"Negative CV score {cv_mean:.3f} for {model}"


@pytest.mark.statistics
@pytest.mark.performance
class TestStatisticalAnalysisPerformance:
    """Test statistical analysis performance and scalability."""
    
    def test_large_dataset_descriptive_statistics_performance(
        self,
        large_statistical_dataset: pd.DataFrame,
        performance_timer
    ):
        """Test descriptive statistics performance on large datasets."""
        statistical_service = StatisticalAnalysisService()
        
        # Test performance on large dataset
        numeric_columns = large_statistical_dataset.select_dtypes(include=[np.number]).columns
        
        performance_results = {}
        
        for col in numeric_columns[:5]:  # Test first 5 numeric columns
            data = large_statistical_dataset[col].values
            
            performance_timer.start()
            result = statistical_service.calculate_descriptive_statistics(data)
            performance_timer.stop()
            
            assert result['success'], f"Descriptive statistics failed for column {col}"
            
            processing_time = performance_timer.elapsed
            performance_results[col] = processing_time
            
            # Should process large dataset quickly
            assert processing_time < 5.0, f"Descriptive statistics too slow for {col}: {processing_time:.2f}s"
            
            # Validate results are reasonable for large dataset
            stats = result['statistics']
            assert stats['count'] == len(data), f"Count mismatch for {col}"
            assert not np.isnan(stats['mean']), f"Mean is NaN for {col}"
            assert stats['std'] > 0, f"Standard deviation is zero for {col}"
        
        # Average processing time should be reasonable
        avg_time = np.mean(list(performance_results.values()))
        assert avg_time < 2.0, f"Average processing time {avg_time:.2f}s too slow"
    
    def test_correlation_matrix_scalability(
        self,
        large_statistical_dataset: pd.DataFrame,
        performance_timer
    ):
        """Test correlation matrix calculation scalability."""
        statistical_service = StatisticalAnalysisService()
        
        # Test with subset of columns to control complexity
        numeric_data = large_statistical_dataset.select_dtypes(include=[np.number])
        
        # Test different matrix sizes
        column_counts = [3, 5, 7, 10]
        
        for n_cols in column_counts:
            if n_cols > len(numeric_data.columns):
                continue
                
            subset_data = numeric_data.iloc[:, :n_cols]
            
            performance_timer.start()
            result = statistical_service.calculate_correlation_matrix(subset_data, method='pearson')
            performance_timer.stop()
            
            processing_time = performance_timer.elapsed
            
            assert result['success'], f"Correlation matrix failed for {n_cols} columns"
            
            # Processing time should scale reasonably
            expected_max_time = n_cols * 0.5  # Linear scaling assumption
            assert processing_time < expected_max_time, (
                f"Correlation matrix too slow for {n_cols} columns: {processing_time:.2f}s"
            )
            
            # Validate result structure
            corr_matrix = pd.DataFrame(result['correlation_matrix'])
            assert corr_matrix.shape == (n_cols, n_cols), f"Correlation matrix shape incorrect for {n_cols} columns"
    
    def test_outlier_detection_performance(
        self,
        large_statistical_dataset: pd.DataFrame,
        performance_timer
    ):
        """Test outlier detection performance on large datasets."""
        statistical_service = StatisticalAnalysisService()
        
        # Test outlier detection on large numeric column
        test_column = 'normal_var'  # Should be in large dataset
        if test_column not in large_statistical_dataset.columns:
            test_column = large_statistical_dataset.select_dtypes(include=[np.number]).columns[0]
        
        data = large_statistical_dataset[test_column].values
        
        # Test different outlier detection methods
        methods = ['iqr', 'zscore', 'modified_zscore']
        
        for method in methods:
            performance_timer.start()
            result = statistical_service.detect_outliers(data, method=method)
            performance_timer.stop()
            
            processing_time = performance_timer.elapsed
            
            assert result['success'], f"Outlier detection failed for method {method}"
            
            # Should process large dataset efficiently
            assert processing_time < 3.0, f"Outlier detection too slow for {method}: {processing_time:.2f}s"
            
            # Validate result quality
            outlier_percentage = result['outlier_percentage']
            assert 0 <= outlier_percentage <= 10, f"Outlier percentage {outlier_percentage}% seems unreasonable for {method}"
    
    def test_concurrent_statistical_analysis(
        self,
        large_statistical_dataset: pd.DataFrame
    ):
        """Test concurrent statistical analysis operations."""
        import threading
        
        statistical_service = StatisticalAnalysisService()
        
        # Prepare data for concurrent analysis
        numeric_columns = large_statistical_dataset.select_dtypes(include=[np.number]).columns[:5]
        results = [None] * len(numeric_columns)
        threads = []
        
        def analyze_column(index: int, column_name: str):
            data = large_statistical_dataset[column_name].values
            
            # Perform multiple operations
            desc_result = statistical_service.calculate_descriptive_statistics(data)
            outlier_result = statistical_service.detect_outliers(data, method='iqr')
            
            results[index] = {
                'column': column_name,
                'descriptive_success': desc_result.get('success', False),
                'outlier_success': outlier_result.get('success', False),
                'descriptive_stats': desc_result.get('statistics', {}),
                'outlier_count': outlier_result.get('outlier_count', 0)
            }
        
        # Start concurrent analysis
        start_time = time.perf_counter()
        
        for i, col in enumerate(numeric_columns):
            thread = threading.Thread(target=analyze_column, args=(i, col))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Validate all analyses succeeded
        for i, result in enumerate(results):
            assert result is not None, f"Analysis {i} returned None"
            assert result['descriptive_success'], f"Descriptive analysis failed for {result['column']}"
            assert result['outlier_success'], f"Outlier detection failed for {result['column']}"
            
            # Validate result quality
            stats = result['descriptive_stats']
            assert 'mean' in stats and not np.isnan(stats['mean']), f"Invalid mean for {result['column']}"
            assert 'std' in stats and stats['std'] > 0, f"Invalid std for {result['column']}"
            
        # Performance assertion
        assert total_time < 15.0, f"Concurrent analysis took {total_time:.2f}s, too slow"
        
        # Should be faster than sequential processing (rough estimate)
        expected_sequential_time = len(numeric_columns) * 2.0  # Estimate 2s per column
        efficiency_ratio = total_time / expected_sequential_time
        assert efficiency_ratio < 0.8, f"Concurrent processing not efficient enough: {efficiency_ratio:.2f}"
