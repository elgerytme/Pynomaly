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
        
        # Advanced distribution analysis
        percentiles = self._calculate_percentiles(clean_series)
        entropy = self._calculate_entropy(clean_series)
        
        # Seasonality detection for time series
        seasonality = self._detect_seasonality(clean_series)
        
        # Drift detection
        drift_analysis = self._detect_drift(clean_series)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_test': normality_test,
            'best_distribution': best_distribution,
            'is_normal': normality_test.get('is_normal', False),
            'percentiles': percentiles,
            'entropy': entropy,
            'seasonality': seasonality,
            'drift_analysis': drift_analysis
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
    
    def _calculate_percentiles(self, series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive percentiles for distribution analysis."""
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        result = {}
        
        for p in percentiles:
            result[f'p{p}'] = float(series.quantile(p / 100))
        
        return result
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy of the distribution."""
        try:
            # Create histogram
            hist, _ = np.histogram(series, bins=min(50, len(series) // 10))
            hist = hist[hist > 0]  # Remove empty bins
            
            # Calculate probabilities
            probabilities = hist / np.sum(hist)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return float(entropy)
        except Exception:
            return 0.0
    
    def _detect_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Detect potential seasonality in numeric data."""
        if len(series) < 24:  # Need sufficient data for seasonality
            return {'has_seasonality': False, 'reason': 'Insufficient data'}
        
        try:
            # Simple autocorrelation-based seasonality detection
            autocorr_12 = series.autocorr(lag=12) if len(series) >= 24 else 0
            autocorr_24 = series.autocorr(lag=24) if len(series) >= 48 else 0
            
            # Check for strong autocorrelation at seasonal lags
            strong_seasonal = abs(autocorr_12) > 0.5 or abs(autocorr_24) > 0.5
            
            return {
                'has_seasonality': strong_seasonal,
                'autocorr_12': float(autocorr_12),
                'autocorr_24': float(autocorr_24),
                'seasonal_strength': max(abs(autocorr_12), abs(autocorr_24))
            }
        except Exception as e:
            return {'has_seasonality': False, 'error': str(e)}
    
    def _detect_drift(self, series: pd.Series) -> Dict[str, Any]:
        """Detect statistical drift in the data."""
        if len(series) < 20:
            return {'has_drift': False, 'reason': 'Insufficient data'}
        
        try:
            # Split data into first and last halves
            mid_point = len(series) // 2
            first_half = series.iloc[:mid_point]
            second_half = series.iloc[mid_point:]
            
            # Calculate means and standard deviations
            mean1, mean2 = first_half.mean(), second_half.mean()
            std1, std2 = first_half.std(), second_half.std()
            
            # Statistical tests for drift
            from scipy import stats
            
            # T-test for mean difference
            t_stat, p_value_mean = stats.ttest_ind(first_half, second_half)
            
            # F-test for variance difference
            f_stat = std1**2 / std2**2 if std2 > 0 else 1
            
            # Simple drift indicators
            mean_drift = abs(mean2 - mean1) / (abs(mean1) + 1e-10)
            std_drift = abs(std2 - std1) / (abs(std1) + 1e-10)
            
            has_drift = p_value_mean < 0.05 or mean_drift > 0.2 or std_drift > 0.2
            
            return {
                'has_drift': has_drift,
                'mean_drift': float(mean_drift),
                'std_drift': float(std_drift),
                'p_value_mean': float(p_value_mean),
                'first_half_mean': float(mean1),
                'second_half_mean': float(mean2)
            }
        except Exception as e:
            return {'has_drift': False, 'error': str(e)}
    
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
        elif method == 'modified_zscore':
            outliers_info = self._detect_outliers_modified_zscore(clean_series)
        elif method == 'lof':
            outliers_info = self._detect_outliers_lof(clean_series)
        
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
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (more robust)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return {'method': 'Modified Z-Score', 'outlier_count': 0, 'outlier_percentage': 0}
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z_scores) > threshold]
        
        return {
            'method': 'Modified Z-Score',
            'threshold': threshold,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_values': outliers.tolist()[:10],
            'max_modified_zscore': float(np.abs(modified_z_scores).max())
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
    
    def _detect_outliers_lof(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit LOF
            lof = LocalOutlierFactor(contamination=0.1)
            outlier_labels = lof.fit_predict(X)
            
            # Get outliers
            outliers = series[outlier_labels == -1]
            
            return {
                'method': 'Local Outlier Factor',
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
    
    def analyze_time_series(self, series: pd.Series, datetime_col: pd.Series = None) -> Dict[str, Any]:
        """Advanced time series analysis for temporal data."""
        if datetime_col is None or len(series) < 10:
            return {}
        
        try:
            # Create time series DataFrame
            ts_df = pd.DataFrame({
                'datetime': pd.to_datetime(datetime_col),
                'value': series
            }).sort_values('datetime')
            
            # Basic time series properties
            time_span = (ts_df['datetime'].max() - ts_df['datetime'].min()).days
            frequency = len(ts_df) / max(time_span, 1)
            
            # Trend analysis
            trend = self._analyze_trend(ts_df)
            
            # Volatility analysis
            volatility = self._analyze_volatility(ts_df)
            
            # Stationarity test
            stationarity = self._test_stationarity(ts_df['value'])
            
            return {
                'time_span_days': time_span,
                'frequency': frequency,
                'trend': trend,
                'volatility': volatility,
                'stationarity': stationarity
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_trend(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend in time series data."""
        try:
            from scipy import stats
            
            # Create numeric time index
            ts_df = ts_df.copy()
            ts_df['time_numeric'] = (ts_df['datetime'] - ts_df['datetime'].min()).dt.total_seconds()
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                ts_df['time_numeric'], ts_df['value']
            )
            
            # Determine trend direction
            if p_value < 0.05:
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'no_trend'
            
            return {
                'direction': trend_direction,
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_volatility(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility in time series data."""
        try:
            values = ts_df['value'].dropna()
            
            # Calculate returns (percentage change)
            returns = values.pct_change().dropna()
            
            # Volatility metrics
            volatility = returns.std()
            mean_return = returns.mean()
            
            # Risk metrics
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Volatility clustering (GARCH effect)
            squared_returns = returns**2
            volatility_clustering = squared_returns.autocorr(lag=1)
            
            return {
                'volatility': float(volatility),
                'mean_return': float(mean_return),
                'sharpe_ratio': float(sharpe_ratio),
                'volatility_clustering': float(volatility_clustering) if not np.isnan(volatility_clustering) else 0,
                'risk_level': 'high' if volatility > 0.1 else 'medium' if volatility > 0.05 else 'low'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Perform ADF test
            result = adfuller(series.dropna())
            
            # Extract results
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Determine stationarity
            is_stationary = p_value < 0.05
            
            return {
                'is_stationary': is_stationary,
                'adf_statistic': float(adf_statistic),
                'p_value': float(p_value),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
            }
        except ImportError:
            return {'error': 'statsmodels not available'}
        except Exception as e:
            return {'error': str(e)}