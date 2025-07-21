"""
Time Series Anomaly Detection and Forecasting for Pynomaly Detection
=====================================================================

Advanced time series analysis with forecasting capabilities, seasonal decomposition,
and specialized algorithms for temporal anomaly detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Import our existing services
from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis."""
    # Detection parameters
    contamination: float = 0.1
    window_size: int = 50
    step_size: int = 1
    
    # Seasonality detection
    detect_seasonality: bool = True
    seasonal_periods: List[int] = field(default_factory=lambda: [24, 168, 8760])  # hourly, weekly, yearly
    min_seasonal_strength: float = 0.3
    
    # Forecasting parameters
    forecast_horizon: int = 24
    confidence_interval: float = 0.95
    
    # Preprocessing
    detrend: bool = True
    deseason: bool = True
    normalize: bool = True
    fill_missing: str = "interpolate"  # interpolate, forward_fill, backward_fill, drop
    
    # Algorithm selection
    detection_algorithm: str = "isolation_forest"  # isolation_forest, lof, prophet_residuals
    forecasting_algorithm: str = "prophet"  # prophet, arima, exp_smoothing, lstm
    
    # Advanced options
    use_external_regressors: bool = False
    detect_changepoints: bool = True
    outlier_removal_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Performance
    parallel_processing: bool = True
    chunk_size: int = 10000

@dataclass
class TimeSeriesResult:
    """Result of time series analysis."""
    timestamps: np.ndarray
    values: np.ndarray
    anomalies: np.ndarray
    anomaly_scores: np.ndarray
    forecasts: Optional[np.ndarray] = None
    forecast_intervals: Optional[Dict[str, np.ndarray]] = None
    seasonal_components: Optional[Dict[str, np.ndarray]] = None
    trend_components: Optional[np.ndarray] = None
    changepoints: Optional[List[datetime]] = None
    seasonality_strength: Optional[float] = None
    stationarity_test: Optional[Dict[str, float]] = None
    model_metrics: Optional[Dict[str, float]] = None

class TimeSeriesDetector:
    """Advanced time series anomaly detector with forecasting capabilities."""
    
    def __init__(self, config: TimeSeriesConfig = None):
        """Initialize time series detector.
        
        Args:
            config: Time series configuration
        """
        self.config = config or TimeSeriesConfig()
        self.core_service = CoreDetectionService()
        self.scaler = None
        self.seasonal_components = {}
        self.trend_model = None
        self.forecast_model = None
        self.is_fitted = False
        
        logger.info(f"Time Series Detector initialized with {self.config.detection_algorithm} detection")
    
    def fit_detect(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                   timestamp_col: Optional[str] = None,
                   value_col: Optional[str] = None) -> TimeSeriesResult:
        """Fit detector and perform anomaly detection on time series data.
        
        Args:
            data: Time series data
            timestamp_col: Name of timestamp column (if DataFrame)
            value_col: Name of value column (if DataFrame)
            
        Returns:
            Time series analysis result
        """
        # Prepare data
        timestamps, values = self._prepare_data(data, timestamp_col, value_col)
        
        # Preprocess data
        processed_values = self._preprocess_data(values, timestamps)
        
        # Detect seasonality and decompose
        seasonal_info = self._analyze_seasonality(processed_values, timestamps)
        
        # Detect anomalies
        anomalies, anomaly_scores = self._detect_anomalies(processed_values, timestamps)
        
        # Generate forecasts
        forecasts, forecast_intervals = self._generate_forecasts(processed_values, timestamps)
        
        # Detect changepoints
        changepoints = self._detect_changepoints(processed_values, timestamps)
        
        # Test stationarity
        stationarity_test = self._test_stationarity(processed_values)
        
        # Compute metrics
        model_metrics = self._compute_metrics(processed_values, anomalies)
        
        self.is_fitted = True
        
        result = TimeSeriesResult(
            timestamps=timestamps,
            values=values,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            forecasts=forecasts,
            forecast_intervals=forecast_intervals,
            seasonal_components=seasonal_info.get('components'),
            trend_components=seasonal_info.get('trend'),
            changepoints=changepoints,
            seasonality_strength=seasonal_info.get('strength'),
            stationarity_test=stationarity_test,
            model_metrics=model_metrics
        )
        
        logger.info(f"Time series analysis completed: {len(anomalies[anomalies==1])} anomalies detected")
        return result
    
    def predict(self, data: Union[pd.DataFrame, pd.Series, np.ndarray],
                timestamp_col: Optional[str] = None,
                value_col: Optional[str] = None) -> TimeSeriesResult:
        """Predict anomalies on new time series data.
        
        Args:
            data: New time series data
            timestamp_col: Name of timestamp column (if DataFrame)
            value_col: Name of value column (if DataFrame)
            
        Returns:
            Time series prediction result
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Prepare data
        timestamps, values = self._prepare_data(data, timestamp_col, value_col)
        
        # Preprocess data
        processed_values = self._preprocess_data(values, timestamps, fit=False)
        
        # Detect anomalies
        anomalies, anomaly_scores = self._detect_anomalies(processed_values, timestamps)
        
        # Generate forecasts if model is available
        forecasts, forecast_intervals = None, None
        if self.forecast_model is not None:
            forecasts, forecast_intervals = self._generate_forecasts(processed_values, timestamps)
        
        return TimeSeriesResult(
            timestamps=timestamps,
            values=values,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            forecasts=forecasts,
            forecast_intervals=forecast_intervals
        )
    
    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray],
                     timestamp_col: Optional[str] = None,
                     value_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate time series data."""
        if isinstance(data, pd.DataFrame):
            if timestamp_col is None or value_col is None:
                raise ValueError("timestamp_col and value_col must be specified for DataFrame")
            
            # Sort by timestamp
            data = data.sort_values(timestamp_col).reset_index(drop=True)
            timestamps = pd.to_datetime(data[timestamp_col]).values
            values = data[value_col].values
            
        elif isinstance(data, pd.Series):
            if hasattr(data.index, 'to_pydatetime'):
                timestamps = data.index.to_pydatetime()
            else:
                timestamps = np.arange(len(data))
            values = data.values
            
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                timestamps = np.arange(len(data))
                values = data
            elif data.ndim == 2 and data.shape[1] == 2:
                timestamps = data[:, 0]
                values = data[:, 1]
            else:
                raise ValueError("Unsupported array shape for time series data")
        else:
            raise ValueError("Data must be DataFrame, Series, or ndarray")
        
        # Handle missing values
        if self.config.fill_missing != "drop":
            mask = ~np.isnan(values)
            if not np.all(mask):
                if self.config.fill_missing == "interpolate":
                    values = pd.Series(values).interpolate().values
                elif self.config.fill_missing == "forward_fill":
                    values = pd.Series(values).fillna(method='ffill').values
                elif self.config.fill_missing == "backward_fill":
                    values = pd.Series(values).fillna(method='bfill').values
        else:
            mask = ~np.isnan(values)
            timestamps = timestamps[mask]
            values = values[mask]
        
        return timestamps, values
    
    def _preprocess_data(self, values: np.ndarray, timestamps: np.ndarray, fit: bool = True) -> np.ndarray:
        """Preprocess time series data."""
        processed_values = values.copy()
        
        # Remove outliers if specified
        if self.config.outlier_removal_method and fit:
            processed_values = self._remove_outliers(processed_values)
        
        # Normalization
        if self.config.normalize:
            if fit:
                self.scaler = StandardScaler()
                processed_values = self.scaler.fit_transform(processed_values.reshape(-1, 1)).flatten()
            else:
                if self.scaler is not None:
                    processed_values = self.scaler.transform(processed_values.reshape(-1, 1)).flatten()
        
        return processed_values
    
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove outliers using specified method."""
        if self.config.outlier_removal_method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (values >= lower_bound) & (values <= upper_bound)
            
            # Replace outliers with median
            median_val = np.median(values)
            values[~mask] = median_val
            
        elif self.config.outlier_removal_method == "zscore":
            z_scores = np.abs(stats.zscore(values))
            mask = z_scores < 3
            
            # Replace outliers with median
            median_val = np.median(values)
            values[~mask] = median_val
            
        elif self.config.outlier_removal_method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(values.reshape(-1, 1))
            mask = outlier_labels == 1
            
            # Replace outliers with median
            median_val = np.median(values)
            values[~mask] = median_val
        
        return values
    
    def _analyze_seasonality(self, values: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonality in time series data."""
        seasonal_info = {}
        
        if not self.config.detect_seasonality or not STATSMODELS_AVAILABLE:
            return seasonal_info
        
        try:
            # Convert to pandas series for statsmodels
            ts_series = pd.Series(values, index=pd.to_datetime(timestamps))
            
            # Test different seasonal periods
            best_strength = 0
            best_period = None
            best_components = None
            
            for period in self.config.seasonal_periods:
                if len(values) < 2 * period:
                    continue
                
                try:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        ts_series, 
                        model='additive', 
                        period=period, 
                        extrapolate_trend='freq'
                    )
                    
                    # Compute seasonal strength
                    seasonal_strength = self._compute_seasonal_strength(
                        decomposition.seasonal.values,
                        decomposition.resid.values
                    )
                    
                    if seasonal_strength > best_strength:
                        best_strength = seasonal_strength
                        best_period = period
                        best_components = {
                            'seasonal': decomposition.seasonal.values,
                            'trend': decomposition.trend.values,
                            'residual': decomposition.resid.values
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to decompose with period {period}: {e}")
                    continue
            
            if best_strength > self.config.min_seasonal_strength:
                seasonal_info = {
                    'strength': best_strength,
                    'period': best_period,
                    'components': best_components,
                    'trend': best_components['trend'] if best_components else None
                }
                
                # Store for later use
                self.seasonal_components = seasonal_info
                
                logger.info(f"Seasonality detected: period={best_period}, strength={best_strength:.3f}")
            
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
        
        return seasonal_info
    
    def _compute_seasonal_strength(self, seasonal: np.ndarray, residual: np.ndarray) -> float:
        """Compute seasonal strength metric."""
        try:
            # Remove NaN values
            mask = ~(np.isnan(seasonal) | np.isnan(residual))
            seasonal_clean = seasonal[mask]
            residual_clean = residual[mask]
            
            if len(seasonal_clean) == 0:
                return 0.0
            
            # Compute seasonal strength as ratio of seasonal variance to residual variance
            seasonal_var = np.var(seasonal_clean)
            residual_var = np.var(residual_clean)
            
            if residual_var == 0:
                return 1.0 if seasonal_var > 0 else 0.0
            
            strength = seasonal_var / (seasonal_var + residual_var)
            return strength
            
        except Exception as e:
            logger.warning(f"Failed to compute seasonal strength: {e}")
            return 0.0
    
    def _detect_anomalies(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in time series data."""
        if self.config.detection_algorithm == "isolation_forest":
            return self._detect_with_isolation_forest(values)
        elif self.config.detection_algorithm == "lof":
            return self._detect_with_lof(values)
        elif self.config.detection_algorithm == "prophet_residuals":
            return self._detect_with_prophet_residuals(values, timestamps)
        else:
            raise ValueError(f"Unknown detection algorithm: {self.config.detection_algorithm}")
    
    def _detect_with_isolation_forest(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Isolation Forest with sliding window."""
        anomalies = np.zeros(len(values), dtype=int)
        anomaly_scores = np.zeros(len(values))
        
        # Use sliding window approach for time series
        for i in range(len(values)):
            start_idx = max(0, i - self.config.window_size + 1)
            end_idx = i + 1
            
            window_data = values[start_idx:end_idx].reshape(-1, 1)
            
            if len(window_data) < 10:  # Need minimum samples
                continue
            
            # Fit Isolation Forest on window
            iso_forest = IsolationForest(
                contamination=self.config.contamination,
                random_state=42,
                n_estimators=100
            )
            
            try:
                labels = iso_forest.fit_predict(window_data)
                scores = iso_forest.decision_function(window_data)
                
                # Mark current point
                current_label = labels[-1]
                current_score = scores[-1]
                
                anomalies[i] = 1 if current_label == -1 else 0
                anomaly_scores[i] = -current_score  # Negative because IF returns negative scores for anomalies
                
            except Exception as e:
                logger.warning(f"Isolation Forest failed at index {i}: {e}")
                continue
        
        return anomalies, anomaly_scores
    
    def _detect_with_lof(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Local Outlier Factor with sliding window."""
        from sklearn.neighbors import LocalOutlierFactor
        
        anomalies = np.zeros(len(values), dtype=int)
        anomaly_scores = np.zeros(len(values))
        
        # Use sliding window approach
        for i in range(len(values)):
            start_idx = max(0, i - self.config.window_size + 1)
            end_idx = i + 1
            
            window_data = values[start_idx:end_idx].reshape(-1, 1)
            
            if len(window_data) < 10:
                continue
            
            # Fit LOF on window
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(window_data) - 1),
                contamination=self.config.contamination,
                novelty=False
            )
            
            try:
                labels = lof.fit_predict(window_data)
                scores = lof.negative_outlier_factor_
                
                # Mark current point
                current_label = labels[-1]
                current_score = scores[-1]
                
                anomalies[i] = 1 if current_label == -1 else 0
                anomaly_scores[i] = -current_score
                
            except Exception as e:
                logger.warning(f"LOF failed at index {i}: {e}")
                continue
        
        return anomalies, anomaly_scores
    
    def _detect_with_prophet_residuals(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Prophet model residuals."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, falling back to Isolation Forest")
            return self._detect_with_isolation_forest(values)
        
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': pd.to_datetime(timestamps),
                'y': values
            })
            
            # Fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)
            
            # Generate predictions
            predictions = model.predict(df)
            
            # Compute residuals
            residuals = values - predictions['yhat'].values
            
            # Detect anomalies in residuals using statistical methods
            residual_std = np.std(residuals)
            threshold = 2.5 * residual_std
            
            anomalies = (np.abs(residuals) > threshold).astype(int)
            anomaly_scores = np.abs(residuals) / residual_std
            
            return anomalies, anomaly_scores
            
        except Exception as e:
            logger.warning(f"Prophet-based detection failed: {e}")
            return self._detect_with_isolation_forest(values)
    
    def _generate_forecasts(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Generate forecasts using specified algorithm."""
        if self.config.forecasting_algorithm == "prophet":
            return self._forecast_with_prophet(values, timestamps)
        elif self.config.forecasting_algorithm == "arima":
            return self._forecast_with_arima(values, timestamps)
        elif self.config.forecasting_algorithm == "exp_smoothing":
            return self._forecast_with_exp_smoothing(values, timestamps)
        else:
            logger.warning(f"Unknown forecasting algorithm: {self.config.forecasting_algorithm}")
            return None, None
    
    def _forecast_with_prophet(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Generate forecasts using Prophet."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return None, None
        
        try:
            # Prepare data
            df = pd.DataFrame({
                'ds': pd.to_datetime(timestamps),
                'y': values
            })
            
            # Fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=self.config.confidence_interval
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)
            
            # Generate future dates
            future_dates = model.make_future_dataframe(periods=self.config.forecast_horizon)
            
            # Make predictions
            forecast = model.predict(future_dates)
            
            # Extract forecasts for future periods
            future_forecasts = forecast['yhat'].iloc[-self.config.forecast_horizon:].values
            future_lower = forecast['yhat_lower'].iloc[-self.config.forecast_horizon:].values
            future_upper = forecast['yhat_upper'].iloc[-self.config.forecast_horizon:].values
            
            forecast_intervals = {
                'lower': future_lower,
                'upper': future_upper
            }
            
            # Store model for future use
            self.forecast_model = model
            
            return future_forecasts, forecast_intervals
            
        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {e}")
            return None, None
    
    def _forecast_with_arima(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Generate forecasts using ARIMA."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available")
            return None, None
        
        try:
            # Auto ARIMA (simple version)
            # In practice, you'd use auto_arima from pmdarima
            model = ARIMA(values, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=self.config.forecast_horizon, alpha=1-self.config.confidence_interval)
            
            forecasts = forecast_result[0]
            conf_int = forecast_result[1]
            
            forecast_intervals = {
                'lower': conf_int[:, 0],
                'upper': conf_int[:, 1]
            }
            
            self.forecast_model = fitted_model
            
            return forecasts, forecast_intervals
            
        except Exception as e:
            logger.warning(f"ARIMA forecasting failed: {e}")
            return None, None
    
    def _forecast_with_exp_smoothing(self, values: np.ndarray, timestamps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Generate forecasts using Exponential Smoothing."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available")
            return None, None
        
        try:
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                values,
                seasonal_periods=24,  # Assume hourly data with daily seasonality
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit()
            
            # Generate forecasts
            forecasts = fitted_model.forecast(steps=self.config.forecast_horizon)
            
            # Approximate confidence intervals (simple method)
            residuals = fitted_model.resid
            residual_std = np.std(residuals[~np.isnan(residuals)])
            z_score = 1.96  # For 95% confidence
            
            forecast_intervals = {
                'lower': forecasts - z_score * residual_std,
                'upper': forecasts + z_score * residual_std
            }
            
            self.forecast_model = fitted_model
            
            return forecasts.values, forecast_intervals
            
        except Exception as e:
            logger.warning(f"Exponential Smoothing forecasting failed: {e}")
            return None, None
    
    def _detect_changepoints(self, values: np.ndarray, timestamps: np.ndarray) -> Optional[List[datetime]]:
        """Detect changepoints in time series."""
        if not self.config.detect_changepoints:
            return None
        
        try:
            # Simple changepoint detection using CUSUM
            changepoints = []
            
            # Compute cumulative sum of deviations from mean
            mean_val = np.mean(values)
            cusum = np.cumsum(values - mean_val)
            
            # Find significant changes in CUSUM
            threshold = 3 * np.std(cusum)
            
            for i in range(1, len(cusum)):
                if abs(cusum[i] - cusum[i-1]) > threshold:
                    changepoints.append(timestamps[i])
            
            return changepoints
            
        except Exception as e:
            logger.warning(f"Changepoint detection failed: {e}")
            return None
    
    def _test_stationarity(self, values: np.ndarray) -> Optional[Dict[str, float]]:
        """Test stationarity of time series."""
        if not STATSMODELS_AVAILABLE:
            return None
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(values)
            
            # KPSS test
            kpss_result = kpss(values)
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'kpss_critical_values': kpss_result[3]
            }
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return None
    
    def _compute_metrics(self, values: np.ndarray, anomalies: np.ndarray) -> Dict[str, float]:
        """Compute model performance metrics."""
        metrics = {}
        
        try:
            # Basic statistics
            metrics['mean'] = np.mean(values)
            metrics['std'] = np.std(values)
            metrics['min'] = np.min(values)
            metrics['max'] = np.max(values)
            
            # Anomaly statistics
            anomaly_rate = np.mean(anomalies)
            metrics['anomaly_rate'] = anomaly_rate
            metrics['num_anomalies'] = int(np.sum(anomalies))
            
            # Time series specific metrics
            if len(values) > 1:
                # First differences
                diffs = np.diff(values)
                metrics['mean_diff'] = np.mean(diffs)
                metrics['std_diff'] = np.std(diffs)
                
                # Volatility (rolling standard deviation)
                if len(values) >= 10:
                    rolling_std = pd.Series(values).rolling(window=10).std()
                    metrics['avg_volatility'] = np.mean(rolling_std[~np.isnan(rolling_std)])
            
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        info = {
            'config': {
                'detection_algorithm': self.config.detection_algorithm,
                'forecasting_algorithm': self.config.forecasting_algorithm,
                'window_size': self.config.window_size,
                'contamination': self.config.contamination,
                'forecast_horizon': self.config.forecast_horizon
            },
            'is_fitted': self.is_fitted,
            'has_seasonal_components': bool(self.seasonal_components),
            'has_forecast_model': self.forecast_model is not None
        }
        
        if self.seasonal_components:
            info['seasonality'] = {
                'strength': self.seasonal_components.get('strength'),
                'period': self.seasonal_components.get('period')
            }
        
        return info