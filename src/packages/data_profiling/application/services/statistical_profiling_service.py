import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from scipy import stats
from ...domain.entities.data_profile import StatisticalSummary


class StatisticalProfilingService:
    """Advanced service to perform comprehensive statistical profiling on numeric data."""
    
    def __init__(self):
        self.confidence_level = 0.95
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, StatisticalSummary]:
        """Compute comprehensive statistical analysis for each numeric column."""
        column_stats = {}
        
        for col in df.select_dtypes(include='number').columns:
            series = df[col].dropna()
            if series.empty:
                continue
            
            statistical_summary = self._analyze_numeric_column(series)
            column_stats[col] = statistical_summary
        
        return column_stats
    
    def _analyze_numeric_column(self, series: pd.Series) -> StatisticalSummary:
        """Perform comprehensive statistical analysis on a numeric column."""
        if series.empty:
            return StatisticalSummary()
        
        # Basic descriptive statistics
        min_val = float(series.min())
        max_val = float(series.max())
        mean_val = float(series.mean())
        median_val = float(series.median())
        std_dev = float(series.std())
        
        # Quartiles
        quartiles = series.quantile([0.25, 0.5, 0.75]).tolist()
        
        return StatisticalSummary(
            min_value=min_val,
            max_value=max_val,
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            quartiles=quartiles
        )
    
    def analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution characteristics of a numeric series."""
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            return {}
        
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return {}
        
        # Distribution characteristics
        skewness = float(clean_series.skew())
        kurtosis = float(clean_series.kurtosis())
        
        # Normality test (Shapiro-Wilk for smaller samples, Anderson-Darling for larger)
        normality_test = self._test_normality(clean_series)
        
        # Distribution fitting
        best_distribution = self._fit_distribution(clean_series)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_test': normality_test,
            'best_distribution': best_distribution,
            'is_normal': normality_test.get('is_normal', False)
        }
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test if the series follows a normal distribution."""
        try:
            if len(series) <= 5000:
                # Shapiro-Wilk test for smaller samples
                statistic, p_value = stats.shapiro(series)
                test_name = "Shapiro-Wilk"
            else:
                # Anderson-Darling test for larger samples
                result = stats.anderson(series, dist='norm')
                statistic = result.statistic
                # Use the critical value at 5% significance level
                critical_value = result.critical_values[2]  # 5% level
                p_value = 0.05 if statistic > critical_value else 0.1
                test_name = "Anderson-Darling"
            
            is_normal = p_value > 0.05
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': is_normal,
                'interpretation': 'Normal distribution' if is_normal else 'Non-normal distribution'
            }
        except Exception as e:
            return {
                'test_name': 'Failed',
                'error': str(e),
                'is_normal': False
            }
    
    def _fit_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Find the best-fitting distribution for the data."""
        try:
            # List of distributions to test
            distributions_to_test = [
                stats.norm,      # Normal
                stats.lognorm,   # Log-normal
                stats.expon,     # Exponential
                stats.gamma,     # Gamma
                stats.uniform,   # Uniform
                stats.beta,      # Beta (if data is between 0 and 1)
            ]
            
            best_distribution = None
            best_params = None
            best_ks_stat = float('inf')
            best_p_value = 0
            
            for distribution in distributions_to_test:
                try:
                    # Fit the distribution
                    params = distribution.fit(series)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.kstest(series, lambda x: distribution.cdf(x, *params))
                    
                    # Choose distribution with highest p-value (best fit)
                    if p_value > best_p_value:
                        best_distribution = distribution.name
                        best_params = params
                        best_ks_stat = ks_stat
                        best_p_value = p_value
                        
                except Exception:
                    continue
            
            if best_distribution:
                return {
                    'distribution': best_distribution,
                    'parameters': list(best_params) if best_params else [],
                    'ks_statistic': float(best_ks_stat),
                    'p_value': float(best_p_value),
                    'goodness_of_fit': 'Good' if best_p_value > 0.05 else 'Poor'
                }
            else:
                return {'distribution': 'Unknown', 'goodness_of_fit': 'Failed to fit'}
                
        except Exception as e:
            return {'distribution': 'Error', 'error': str(e)}
    
    def detect_outliers(self, series: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers using various methods."""
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            return {}
        
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return {}
        
        outliers_info = {}
        
        if method == 'iqr':
            outliers_info = self._detect_outliers_iqr(clean_series)
        elif method == 'zscore':
            outliers_info = self._detect_outliers_zscore(clean_series)
        elif method == 'isolation_forest':
            outliers_info = self._detect_outliers_isolation_forest(clean_series)
        
        return outliers_info
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'method': 'IQR',
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_values': outliers.tolist()[:10]  # Limit to first 10
        }
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outliers = series[z_scores > threshold]
        
        return {
            'method': 'Z-Score',
            'threshold': threshold,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_values': outliers.tolist()[:10],
            'max_zscore': float(z_scores.max())
        }
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest (simplified version)."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Get outliers
            outliers = series[outlier_labels == -1]
            
            return {
                'method': 'Isolation Forest',
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(series)) * 100,
                'outlier_values': outliers.tolist()[:10]
            }
        except ImportError:
            # Fallback to IQR if sklearn not available
            return self._detect_outliers_iqr(series)
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis on numeric columns."""
        numeric_df = df.select_dtypes(include='number')
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'Strong' if abs(corr_value) > 0.8 else 'Moderate'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.round(3).to_dict(),
            'high_correlations': high_correlations,
            'max_correlation': float(correlation_matrix.max().max()),
            'min_correlation': float(correlation_matrix.min().min())
        }