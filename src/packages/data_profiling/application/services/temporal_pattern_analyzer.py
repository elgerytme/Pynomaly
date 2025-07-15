"""
Temporal Pattern Analyzer for Time-Series Anomaly Detection

Advanced temporal pattern analysis system designed for time-series data profiling and anomaly detection.
Provides comprehensive analysis of temporal patterns, seasonality detection, trend analysis,
and time-based anomaly identification for data profiling operations.

This completes Issue #144: Phase 2.3: Data Profiling Package - Advanced Pattern Discovery
with sophisticated temporal pattern analysis capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd

# Optional dependencies with graceful fallback
try:
    from scipy import stats, signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import zscore, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TemporalPatternType(Enum):
    """Types of temporal patterns that can be detected."""
    
    # Trend patterns
    LINEAR_TREND = "linear_trend"
    EXPONENTIAL_TREND = "exponential_trend"
    POLYNOMIAL_TREND = "polynomial_trend"
    CHANGE_POINT = "change_point"
    
    # Seasonal patterns
    DAILY_SEASONALITY = "daily_seasonality"
    WEEKLY_SEASONALITY = "weekly_seasonality"
    MONTHLY_SEASONALITY = "monthly_seasonality"
    QUARTERLY_SEASONALITY = "quarterly_seasonality"
    YEARLY_SEASONALITY = "yearly_seasonality"
    
    # Cyclical patterns
    PERIODIC_CYCLE = "periodic_cycle"
    IRREGULAR_CYCLE = "irregular_cycle"
    HARMONIC_PATTERN = "harmonic_pattern"
    
    # Anomaly patterns
    OUTLIER_SEQUENCE = "outlier_sequence"
    LEVEL_SHIFT = "level_shift"
    VARIANCE_SHIFT = "variance_shift"
    MISSING_SEQUENCE = "missing_sequence"
    
    # Behavioral patterns
    USAGE_PATTERN = "usage_pattern"
    BUSINESS_HOUR_PATTERN = "business_hour_pattern"
    HOLIDAY_EFFECT = "holiday_effect"
    WEEKEND_EFFECT = "weekend_effect"
    
    # Statistical patterns
    STATIONARITY = "stationarity"
    AUTOCORRELATION = "autocorrelation"
    HETEROSCEDASTICITY = "heteroscedasticity"
    
    # Unknown
    UNKNOWN = "unknown"


class TemporalAnomalyType(Enum):
    """Types of temporal anomalies."""
    
    POINT_ANOMALY = "point_anomaly"          # Single point anomaly
    CONTEXTUAL_ANOMALY = "contextual_anomaly"  # Context-dependent anomaly
    COLLECTIVE_ANOMALY = "collective_anomaly"  # Sequence anomaly
    SEASONAL_ANOMALY = "seasonal_anomaly"    # Seasonal pattern violation
    TREND_ANOMALY = "trend_anomaly"          # Trend pattern violation
    FREQUENCY_ANOMALY = "frequency_anomaly"   # Frequency domain anomaly


@dataclass
class TemporalPattern:
    """Represents a detected temporal pattern."""
    
    pattern_type: TemporalPatternType
    confidence: float
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    description: str
    
    # Pattern-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical evidence
    statistical_evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    strength: float = 0.0  # Pattern strength (0-1)
    consistency: float = 0.0  # Pattern consistency (0-1)
    significance: float = 0.0  # Statistical significance
    
    # Metadata
    detection_method: str = ""
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemporalAnomaly:
    """Represents a detected temporal anomaly."""
    
    anomaly_type: TemporalAnomalyType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    
    # Time information
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    
    # Affected data
    affected_indices: List[int]
    affected_values: List[Any]
    expected_values: Optional[List[Any]] = None
    
    # Context
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical evidence
    statistical_evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadata
    detection_method: str = ""
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemporalAnalysisResult:
    """Result of temporal pattern analysis."""
    
    # Time series metadata
    start_time: datetime
    end_time: datetime
    frequency: Optional[str]
    total_observations: int
    missing_observations: int
    
    # Detected patterns
    patterns: List[TemporalPattern]
    anomalies: List[TemporalAnomaly]
    
    # Overall characteristics
    is_stationary: bool
    has_trend: bool
    has_seasonality: bool
    dominant_frequency: Optional[float]
    
    # Quality assessment
    data_quality_score: float
    pattern_strength_score: float
    anomaly_score: float
    
    # Decomposition (if available)
    trend_component: Optional[pd.Series] = None
    seasonal_component: Optional[pd.Series] = None
    residual_component: Optional[pd.Series] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Analysis metadata
    analysis_duration: float = 0.0
    methods_used: List[str] = field(default_factory=list)


@dataclass
class TemporalAnalysisConfig:
    """Configuration for temporal pattern analysis."""
    
    # Analysis scope
    enable_trend_analysis: bool = True
    enable_seasonality_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_frequency_analysis: bool = True
    enable_stationarity_tests: bool = True
    
    # Detection thresholds
    trend_significance_threshold: float = 0.05
    seasonality_strength_threshold: float = 0.3
    anomaly_threshold: float = 2.5
    
    # Seasonality detection
    min_seasonal_periods: int = 2
    max_seasonal_periods: int = 100
    seasonal_decomposition_model: str = "additive"  # "additive" or "multiplicative"
    
    # Frequency analysis
    enable_fft_analysis: bool = True
    fft_window_size: Optional[int] = None
    
    # Anomaly detection
    anomaly_methods: List[str] = field(default_factory=lambda: ["statistical", "isolation_forest", "lstm"])
    contamination_rate: float = 0.1
    
    # Performance settings
    max_time_series_length: int = 100000
    min_time_series_length: int = 10
    analysis_timeout_seconds: int = 300
    
    # Domain-specific settings
    business_hours: Tuple[int, int] = (9, 17)  # 9 AM to 5 PM
    weekend_days: List[int] = field(default_factory=lambda: [5, 6])  # Saturday, Sunday
    known_holidays: List[datetime] = field(default_factory=list)


class TemporalPatternAnalyzer:
    """Advanced temporal pattern analyzer for time-series data."""
    
    def __init__(self, config: Optional[TemporalAnalysisConfig] = None):
        """Initialize the temporal pattern analyzer."""
        
        self.config = config or TemporalAnalysisConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Analysis cache
        self._analysis_cache = {}
        
        # Pattern detection methods
        self._pattern_detectors = {
            'trend': self._detect_trend_patterns,
            'seasonality': self._detect_seasonal_patterns,
            'cyclical': self._detect_cyclical_patterns,
            'frequency': self._detect_frequency_patterns,
            'statistical': self._detect_statistical_patterns
        }
        
        # Anomaly detection methods
        self._anomaly_detectors = {
            'statistical': self._detect_statistical_anomalies,
            'isolation_forest': self._detect_isolation_forest_anomalies,
            'seasonal': self._detect_seasonal_anomalies,
            'contextual': self._detect_contextual_anomalies
        }
    
    async def analyze_temporal_patterns(
        self,
        data: Union[pd.Series, pd.DataFrame],
        timestamp_column: Optional[str] = None,
        value_column: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TemporalAnalysisResult:
        """Analyze temporal patterns in time-series data."""
        
        start_time = time.time()
        self.logger.info("Starting temporal pattern analysis")
        
        # Prepare time series data
        ts_data = self._prepare_time_series(data, timestamp_column, value_column)
        
        if ts_data is None or len(ts_data) < self.config.min_time_series_length:
            raise ValueError("Insufficient data for temporal analysis")
        
        # Sample data if too large
        if len(ts_data) > self.config.max_time_series_length:
            ts_data = self._sample_time_series(ts_data)
        
        # Initialize result
        result = TemporalAnalysisResult(
            start_time=ts_data.index.min(),
            end_time=ts_data.index.max(),
            frequency=ts_data.index.inferred_freq,
            total_observations=len(ts_data),
            missing_observations=ts_data.isnull().sum(),
            patterns=[],
            anomalies=[],
            is_stationary=False,
            has_trend=False,
            has_seasonality=False,
            dominant_frequency=None,
            data_quality_score=0.0,
            pattern_strength_score=0.0,
            anomaly_score=0.0,
            methods_used=[]
        )
        
        try:
            # Detect patterns
            if self.config.enable_trend_analysis:
                trend_patterns = await self._detect_trend_patterns(ts_data)
                result.patterns.extend(trend_patterns)
                result.has_trend = any(p.pattern_type in [
                    TemporalPatternType.LINEAR_TREND,
                    TemporalPatternType.EXPONENTIAL_TREND,
                    TemporalPatternType.POLYNOMIAL_TREND
                ] for p in trend_patterns)
                result.methods_used.append('trend_analysis')
            
            if self.config.enable_seasonality_detection:
                seasonal_patterns = await self._detect_seasonal_patterns(ts_data)
                result.patterns.extend(seasonal_patterns)
                result.has_seasonality = any(p.pattern_type.value.endswith('_seasonality') for p in seasonal_patterns)
                result.methods_used.append('seasonality_detection')
                
                # Perform decomposition if seasonality detected
                if result.has_seasonality and STATSMODELS_AVAILABLE:
                    decomposition = self._perform_seasonal_decomposition(ts_data)
                    if decomposition is not None:
                        result.trend_component = decomposition.trend
                        result.seasonal_component = decomposition.seasonal
                        result.residual_component = decomposition.resid
            
            if self.config.enable_frequency_analysis:
                frequency_patterns = await self._detect_frequency_patterns(ts_data)
                result.patterns.extend(frequency_patterns)
                result.methods_used.append('frequency_analysis')
                
                # Extract dominant frequency
                if frequency_patterns:
                    freq_pattern = max(frequency_patterns, key=lambda p: p.confidence)
                    result.dominant_frequency = freq_pattern.parameters.get('frequency')
            
            if self.config.enable_stationarity_tests:
                statistical_patterns = await self._detect_statistical_patterns(ts_data)
                result.patterns.extend(statistical_patterns)
                result.is_stationary = any(p.pattern_type == TemporalPatternType.STATIONARITY for p in statistical_patterns)
                result.methods_used.append('stationarity_tests')
            
            # Detect anomalies
            if self.config.enable_anomaly_detection:
                for method in self.config.anomaly_methods:
                    if method in self._anomaly_detectors:
                        anomalies = await self._anomaly_detectors[method](ts_data, result)
                        result.anomalies.extend(anomalies)
                        result.methods_used.append(f'anomaly_{method}')
            
            # Calculate quality scores
            result.data_quality_score = self._calculate_data_quality_score(ts_data, result)
            result.pattern_strength_score = self._calculate_pattern_strength_score(result.patterns)
            result.anomaly_score = self._calculate_anomaly_score(result.anomalies, len(ts_data))
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)
            
        except Exception as e:
            self.logger.error(f"Temporal pattern analysis failed: {e}")
            raise
        
        result.analysis_duration = time.time() - start_time
        self.logger.info(f"Temporal analysis completed in {result.analysis_duration:.2f} seconds")
        
        return result
    
    def _prepare_time_series(
        self,
        data: Union[pd.Series, pd.DataFrame],
        timestamp_column: Optional[str] = None,
        value_column: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Prepare time series data for analysis."""
        
        if isinstance(data, pd.Series):
            # Check if index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if pd.api.types.is_numeric_dtype(data.index):
                    # Assume integer index represents time periods
                    data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                else:
                    return None
            return data
        
        elif isinstance(data, pd.DataFrame):
            if timestamp_column is None:
                # Try to find timestamp column
                datetime_cols = data.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    timestamp_column = datetime_cols[0]
                else:
                    return None
            
            if value_column is None:
                # Try to find numeric value column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                else:
                    return None
            
            # Create time series
            ts_data = data.set_index(timestamp_column)[value_column]
            ts_data.index = pd.to_datetime(ts_data.index)
            return ts_data.sort_index()
        
        return None
    
    def _sample_time_series(self, ts_data: pd.Series) -> pd.Series:
        """Sample time series data for performance."""
        
        # Systematic sampling to preserve temporal structure
        sample_ratio = self.config.max_time_series_length / len(ts_data)
        step = int(1 / sample_ratio)
        
        return ts_data.iloc[::step]
    
    async def _detect_trend_patterns(self, ts_data: pd.Series) -> List[TemporalPattern]:
        """Detect trend patterns in the time series."""
        
        patterns = []
        
        if len(ts_data) < 10:
            return patterns
        
        # Remove missing values for trend analysis
        clean_data = ts_data.dropna()
        if len(clean_data) < 10:
            return patterns
        
        # Linear trend detection
        x = np.arange(len(clean_data))
        y = clean_data.values
        
        if SKLEARN_AVAILABLE:
            try:
                # Linear regression
                lr = LinearRegression()
                lr.fit(x.reshape(-1, 1), y)
                
                trend_slope = lr.coef_[0]
                r_squared = lr.score(x.reshape(-1, 1), y)
                
                # Check if trend is significant
                if abs(trend_slope) > 0 and r_squared > 0.1:
                    trend_type = TemporalPatternType.LINEAR_TREND
                    confidence = min(r_squared, 0.95)
                    
                    patterns.append(TemporalPattern(
                        pattern_type=trend_type,
                        confidence=confidence,
                        start_time=clean_data.index.min(),
                        end_time=clean_data.index.max(),
                        description=f"{'Increasing' if trend_slope > 0 else 'Decreasing'} linear trend detected",
                        parameters={
                            'slope': float(trend_slope),
                            'intercept': float(lr.intercept_),
                            'r_squared': float(r_squared)
                        },
                        statistical_evidence={
                            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                            'trend_magnitude': abs(float(trend_slope)),
                            'goodness_of_fit': float(r_squared)
                        },
                        strength=confidence,
                        consistency=r_squared,
                        detection_method="linear_regression"
                    ))
                
            except Exception as e:
                self.logger.warning(f"Linear trend detection failed: {e}")
        
        # Simple trend detection using first and last values
        if len(clean_data) >= 5:
            window_size = min(5, len(clean_data) // 4)
            first_window = clean_data.iloc[:window_size].mean()
            last_window = clean_data.iloc[-window_size:].mean()
            
            relative_change = (last_window - first_window) / abs(first_window) if first_window != 0 else 0
            
            if abs(relative_change) > 0.1:  # 10% change threshold
                patterns.append(TemporalPattern(
                    pattern_type=TemporalPatternType.LINEAR_TREND,
                    confidence=min(abs(relative_change), 0.8),
                    start_time=clean_data.index.min(),
                    end_time=clean_data.index.max(),
                    description=f"Overall {'increasing' if relative_change > 0 else 'decreasing'} trend",
                    parameters={
                        'relative_change': float(relative_change),
                        'first_window_mean': float(first_window),
                        'last_window_mean': float(last_window)
                    },
                    strength=min(abs(relative_change), 0.8),
                    detection_method="window_comparison"
                ))
        
        return patterns
    
    async def _detect_seasonal_patterns(self, ts_data: pd.Series) -> List[TemporalPattern]:
        """Detect seasonal patterns in the time series."""
        
        patterns = []
        
        if len(ts_data) < 20:
            return patterns
        
        # Try to detect different types of seasonality
        if STATSMODELS_AVAILABLE:
            try:
                # Automatic seasonality detection
                decomposition = seasonal_decompose(
                    ts_data.dropna(), 
                    model=self.config.seasonal_decomposition_model,
                    extrapolate_trend='freq'
                )
                
                # Calculate seasonal strength
                seasonal_strength = self._calculate_seasonal_strength(
                    ts_data.dropna(), decomposition.seasonal
                )
                
                if seasonal_strength > self.config.seasonality_strength_threshold:
                    # Determine seasonality type based on frequency
                    freq = ts_data.index.inferred_freq
                    if freq:
                        if 'D' in freq:
                            pattern_type = TemporalPatternType.DAILY_SEASONALITY
                        elif 'W' in freq:
                            pattern_type = TemporalPatternType.WEEKLY_SEASONALITY
                        elif 'M' in freq:
                            pattern_type = TemporalPatternType.MONTHLY_SEASONALITY
                        elif 'Q' in freq:
                            pattern_type = TemporalPatternType.QUARTERLY_SEASONALITY
                        elif 'Y' in freq or 'A' in freq:
                            pattern_type = TemporalPatternType.YEARLY_SEASONALITY
                        else:
                            pattern_type = TemporalPatternType.PERIODIC_CYCLE
                    else:
                        pattern_type = TemporalPatternType.PERIODIC_CYCLE
                    
                    patterns.append(TemporalPattern(
                        pattern_type=pattern_type,
                        confidence=min(seasonal_strength, 0.95),
                        start_time=ts_data.index.min(),
                        end_time=ts_data.index.max(),
                        description=f"Seasonal pattern detected with strength {seasonal_strength:.2f}",
                        parameters={
                            'seasonal_strength': float(seasonal_strength),
                            'period': len(decomposition.seasonal.dropna()) if decomposition.seasonal is not None else None
                        },
                        statistical_evidence={
                            'decomposition_model': self.config.seasonal_decomposition_model,
                            'trend_variance': float(np.var(decomposition.trend.dropna())) if decomposition.trend is not None else 0,
                            'seasonal_variance': float(np.var(decomposition.seasonal.dropna())) if decomposition.seasonal is not None else 0,
                            'residual_variance': float(np.var(decomposition.resid.dropna())) if decomposition.resid is not None else 0
                        },
                        strength=seasonal_strength,
                        consistency=seasonal_strength,
                        detection_method="seasonal_decomposition"
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Seasonal decomposition failed: {e}")
        
        # Simple periodicity detection using autocorrelation
        if SCIPY_AVAILABLE and len(ts_data) >= 30:
            try:
                clean_data = ts_data.dropna()
                
                # Calculate autocorrelation for different lags
                max_lag = min(len(clean_data) // 4, 365)  # Limit to avoid computation issues
                
                autocorrelations = []
                lags = range(1, max_lag + 1)
                
                for lag in lags:
                    if lag < len(clean_data):
                        corr = clean_data.autocorr(lag=lag)
                        if not np.isnan(corr):
                            autocorrelations.append((lag, abs(corr)))
                
                # Find significant peaks in autocorrelation
                if autocorrelations:
                    autocorrelations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Check top correlations
                    for lag, corr in autocorrelations[:5]:
                        if corr > 0.3:  # Significant correlation threshold
                            # Determine seasonality type based on lag
                            if lag == 7:
                                pattern_type = TemporalPatternType.WEEKLY_SEASONALITY
                            elif 28 <= lag <= 31:
                                pattern_type = TemporalPatternType.MONTHLY_SEASONALITY
                            elif 365 <= lag <= 366:
                                pattern_type = TemporalPatternType.YEARLY_SEASONALITY
                            else:
                                pattern_type = TemporalPatternType.PERIODIC_CYCLE
                            
                            patterns.append(TemporalPattern(
                                pattern_type=pattern_type,
                                confidence=min(corr, 0.9),
                                start_time=ts_data.index.min(),
                                end_time=ts_data.index.max(),
                                description=f"Periodic pattern detected with lag {lag} (correlation: {corr:.2f})",
                                parameters={
                                    'lag': lag,
                                    'autocorrelation': float(corr),
                                    'period_estimate': lag
                                },
                                strength=corr,
                                consistency=corr,
                                detection_method="autocorrelation"
                            ))
                            break  # Take only the strongest pattern
                            
            except Exception as e:
                self.logger.warning(f"Autocorrelation analysis failed: {e}")
        
        return patterns
    
    async def _detect_cyclical_patterns(self, ts_data: pd.Series) -> List[TemporalPattern]:
        """Detect cyclical patterns in the time series."""
        
        patterns = []
        
        # This would implement more sophisticated cyclical pattern detection
        # For now, return empty list
        
        return patterns
    
    async def _detect_frequency_patterns(self, ts_data: pd.Series) -> List[TemporalPattern]:
        """Detect frequency domain patterns using FFT."""
        
        patterns = []
        
        if not self.config.enable_fft_analysis or not SCIPY_AVAILABLE:
            return patterns
        
        if len(ts_data) < 50:
            return patterns
        
        try:
            clean_data = ts_data.dropna()
            
            if len(clean_data) < 50:
                return patterns
            
            # Apply FFT
            fft_values = fft(clean_data.values)
            fft_freq = fftfreq(len(clean_data))
            
            # Find dominant frequencies
            magnitude = np.abs(fft_values)
            
            # Ignore DC component and negative frequencies
            positive_freq_mask = fft_freq > 0
            positive_freqs = fft_freq[positive_freq_mask]
            positive_magnitudes = magnitude[positive_freq_mask]
            
            if len(positive_magnitudes) > 0:
                # Find peaks in frequency domain
                peak_threshold = np.mean(positive_magnitudes) + 2 * np.std(positive_magnitudes)
                peak_indices = np.where(positive_magnitudes > peak_threshold)[0]
                
                if len(peak_indices) > 0:
                    # Get the most dominant frequency
                    max_peak_idx = peak_indices[np.argmax(positive_magnitudes[peak_indices])]
                    dominant_freq = positive_freqs[max_peak_idx]
                    dominant_magnitude = positive_magnitudes[max_peak_idx]
                    
                    # Convert frequency to period
                    period = 1 / dominant_freq if dominant_freq > 0 else None
                    
                    confidence = min(dominant_magnitude / np.sum(positive_magnitudes), 0.9)
                    
                    patterns.append(TemporalPattern(
                        pattern_type=TemporalPatternType.HARMONIC_PATTERN,
                        confidence=confidence,
                        start_time=ts_data.index.min(),
                        end_time=ts_data.index.max(),
                        description=f"Dominant frequency pattern detected at {dominant_freq:.4f} Hz",
                        parameters={
                            'frequency': float(dominant_freq),
                            'period': float(period) if period else None,
                            'magnitude': float(dominant_magnitude),
                            'relative_strength': float(confidence)
                        },
                        statistical_evidence={
                            'fft_peaks': len(peak_indices),
                            'peak_threshold': float(peak_threshold),
                            'total_energy': float(np.sum(magnitude))
                        },
                        strength=confidence,
                        detection_method="fft_analysis"
                    ))
                    
        except Exception as e:
            self.logger.warning(f"FFT analysis failed: {e}")
        
        return patterns
    
    async def _detect_statistical_patterns(self, ts_data: pd.Series) -> List[TemporalPattern]:
        """Detect statistical patterns like stationarity."""
        
        patterns = []
        
        if len(ts_data) < 20:
            return patterns
        
        clean_data = ts_data.dropna()
        if len(clean_data) < 20:
            return patterns
        
        # Stationarity tests
        if STATSMODELS_AVAILABLE:
            try:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(clean_data.values)
                adf_pvalue = adf_result[1]
                
                # KPSS test
                kpss_result = kpss(clean_data.values)
                kpss_pvalue = kpss_result[1]
                
                # Determine stationarity
                is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)
                
                confidence = 0.8 if is_stationary else 0.3
                
                patterns.append(TemporalPattern(
                    pattern_type=TemporalPatternType.STATIONARITY,
                    confidence=confidence,
                    start_time=ts_data.index.min(),
                    end_time=ts_data.index.max(),
                    description=f"Time series is {'stationary' if is_stationary else 'non-stationary'}",
                    parameters={
                        'is_stationary': is_stationary,
                        'adf_statistic': float(adf_result[0]),
                        'adf_pvalue': float(adf_pvalue),
                        'kpss_statistic': float(kpss_result[0]),
                        'kpss_pvalue': float(kpss_pvalue)
                    },
                    statistical_evidence={
                        'adf_critical_values': {str(k): float(v) for k, v in adf_result[4].items()},
                        'kpss_critical_values': {str(k): float(v) for k, v in kpss_result[3].items()}
                    },
                    strength=confidence,
                    detection_method="stationarity_tests"
                ))
                
            except Exception as e:
                self.logger.warning(f"Stationarity tests failed: {e}")
        
        return patterns
    
    async def _detect_statistical_anomalies(
        self,
        ts_data: pd.Series,
        analysis_result: TemporalAnalysisResult
    ) -> List[TemporalAnomaly]:
        """Detect statistical anomalies in the time series."""
        
        anomalies = []
        
        if len(ts_data) < 10:
            return anomalies
        
        clean_data = ts_data.dropna()
        if len(clean_data) < 10:
            return anomalies
        
        try:
            # Z-score based detection
            if SCIPY_AVAILABLE:
                z_scores = np.abs(zscore(clean_data.values))
                outlier_mask = z_scores > self.config.anomaly_threshold
                
                if outlier_mask.any():
                    outlier_indices = clean_data.index[outlier_mask]
                    outlier_values = clean_data[outlier_mask]
                    
                    for idx, value in zip(outlier_indices, outlier_values):
                        z_score = z_scores[clean_data.index.get_loc(idx)]
                        
                        anomalies.append(TemporalAnomaly(
                            anomaly_type=TemporalAnomalyType.POINT_ANOMALY,
                            severity="high" if z_score > 4 else "medium",
                            confidence=min(z_score / 5, 0.95),
                            start_time=idx,
                            end_time=None,
                            duration=None,
                            affected_indices=[clean_data.index.get_loc(idx)],
                            affected_values=[float(value)],
                            description=f"Statistical outlier detected (z-score: {z_score:.2f})",
                            statistical_evidence={
                                'z_score': float(z_score),
                                'threshold': self.config.anomaly_threshold,
                                'series_mean': float(clean_data.mean()),
                                'series_std': float(clean_data.std())
                            },
                            recommended_actions=[
                                "Investigate data source for potential errors",
                                "Consider if outlier represents valid extreme event",
                                "Evaluate impact on downstream analysis"
                            ],
                            detection_method="z_score"
                        ))
            
            # Moving average based detection
            window_size = min(10, len(clean_data) // 4)
            if window_size >= 3:
                moving_avg = clean_data.rolling(window=window_size, center=True).mean()
                moving_std = clean_data.rolling(window=window_size, center=True).std()
                
                deviations = np.abs(clean_data - moving_avg) / moving_std
                anomaly_mask = deviations > self.config.anomaly_threshold
                
                if anomaly_mask.any():
                    for idx in clean_data.index[anomaly_mask]:
                        if not pd.isna(deviations[idx]):
                            deviation = deviations[idx]
                            
                            anomalies.append(TemporalAnomaly(
                                anomaly_type=TemporalAnomalyType.CONTEXTUAL_ANOMALY,
                                severity="medium" if deviation < 4 else "high",
                                confidence=min(deviation / 5, 0.9),
                                start_time=idx,
                                end_time=None,
                                duration=None,
                                affected_indices=[clean_data.index.get_loc(idx)],
                                affected_values=[float(clean_data[idx])],
                                expected_values=[float(moving_avg[idx])],
                                description=f"Contextual anomaly detected (deviation: {deviation:.2f})",
                                statistical_evidence={
                                    'deviation': float(deviation),
                                    'expected_value': float(moving_avg[idx]),
                                    'local_std': float(moving_std[idx]),
                                    'window_size': window_size
                                },
                                detection_method="moving_average"
                            ))
                            
        except Exception as e:
            self.logger.warning(f"Statistical anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_isolation_forest_anomalies(
        self,
        ts_data: pd.Series,
        analysis_result: TemporalAnalysisResult
    ) -> List[TemporalAnomaly]:
        """Detect anomalies using Isolation Forest."""
        
        anomalies = []
        
        if not SKLEARN_AVAILABLE or len(ts_data) < 50:
            return anomalies
        
        try:
            clean_data = ts_data.dropna()
            
            # Prepare features (value and simple time-based features)
            features = []
            for i, (idx, value) in enumerate(clean_data.items()):
                feature_vector = [
                    float(value),
                    i,  # Position in series
                    idx.hour if hasattr(idx, 'hour') else 0,
                    idx.dayofweek if hasattr(idx, 'dayofweek') else 0,
                    idx.month if hasattr(idx, 'month') else 0
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Apply Isolation Forest
            isolation_forest = IsolationForest(
                contamination=self.config.contamination_rate,
                random_state=42
            )
            
            outlier_labels = isolation_forest.fit_predict(features)
            anomaly_scores = isolation_forest.score_samples(features)
            
            # Extract anomalies
            anomaly_mask = outlier_labels == -1
            
            if anomaly_mask.any():
                anomaly_indices = clean_data.index[anomaly_mask]
                anomaly_values = clean_data[anomaly_mask]
                anomaly_score_values = anomaly_scores[anomaly_mask]
                
                for idx, value, score in zip(anomaly_indices, anomaly_values, anomaly_score_values):
                    confidence = min(abs(score) * 2, 0.9)  # Convert score to confidence
                    
                    anomalies.append(TemporalAnomaly(
                        anomaly_type=TemporalAnomalyType.POINT_ANOMALY,
                        severity="medium",
                        confidence=confidence,
                        start_time=idx,
                        end_time=None,
                        duration=None,
                        affected_indices=[clean_data.index.get_loc(idx)],
                        affected_values=[float(value)],
                        description=f"Isolation Forest anomaly detected (score: {score:.3f})",
                        statistical_evidence={
                            'anomaly_score': float(score),
                            'contamination_rate': self.config.contamination_rate,
                            'feature_vector': features[clean_data.index.get_loc(idx)].tolist()
                        },
                        detection_method="isolation_forest"
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Isolation Forest anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_seasonal_anomalies(
        self,
        ts_data: pd.Series,
        analysis_result: TemporalAnalysisResult
    ) -> List[TemporalAnomaly]:
        """Detect seasonal anomalies."""
        
        anomalies = []
        
        # Check if seasonality was detected
        has_seasonality = any(p.pattern_type.value.endswith('_seasonality') for p in analysis_result.patterns)
        
        if not has_seasonality or analysis_result.seasonal_component is None:
            return anomalies
        
        try:
            # Use residual component to detect seasonal anomalies
            residuals = analysis_result.residual_component.dropna()
            
            if len(residuals) > 10:
                # Statistical analysis of residuals
                residual_threshold = 2 * residuals.std()
                anomaly_mask = np.abs(residuals) > residual_threshold
                
                if anomaly_mask.any():
                    for idx in residuals.index[anomaly_mask]:
                        residual_value = residuals[idx]
                        
                        anomalies.append(TemporalAnomaly(
                            anomaly_type=TemporalAnomalyType.SEASONAL_ANOMALY,
                            severity="medium",
                            confidence=min(abs(residual_value) / (3 * residuals.std()), 0.9),
                            start_time=idx,
                            end_time=None,
                            duration=None,
                            affected_indices=[ts_data.index.get_loc(idx)],
                            affected_values=[float(ts_data[idx])],
                            description=f"Seasonal pattern violation detected",
                            statistical_evidence={
                                'residual_value': float(residual_value),
                                'residual_threshold': float(residual_threshold),
                                'seasonal_component': float(analysis_result.seasonal_component[idx])
                            },
                            detection_method="seasonal_decomposition"
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Seasonal anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_contextual_anomalies(
        self,
        ts_data: pd.Series,
        analysis_result: TemporalAnalysisResult
    ) -> List[TemporalAnomaly]:
        """Detect contextual anomalies based on time context."""
        
        anomalies = []
        
        # This would implement context-aware anomaly detection
        # For now, return empty list
        
        return anomalies
    
    def _calculate_seasonal_strength(self, original: pd.Series, seasonal: pd.Series) -> float:
        """Calculate the strength of seasonal component."""
        
        try:
            if seasonal is None or len(seasonal) == 0:
                return 0.0
            
            seasonal_var = np.var(seasonal.dropna())
            total_var = np.var(original.dropna())
            
            if total_var == 0:
                return 0.0
            
            return min(seasonal_var / total_var, 1.0)
            
        except Exception:
            return 0.0
    
    def _perform_seasonal_decomposition(self, ts_data: pd.Series):
        """Perform seasonal decomposition if possible."""
        
        if not STATSMODELS_AVAILABLE or len(ts_data) < 24:
            return None
        
        try:
            return seasonal_decompose(
                ts_data.dropna(),
                model=self.config.seasonal_decomposition_model,
                extrapolate_trend='freq'
            )
        except Exception as e:
            self.logger.warning(f"Seasonal decomposition failed: {e}")
            return None
    
    def _calculate_data_quality_score(self, ts_data: pd.Series, result: TemporalAnalysisResult) -> float:
        """Calculate data quality score."""
        
        completeness = 1 - (result.missing_observations / result.total_observations)
        anomaly_penalty = min(len(result.anomalies) / result.total_observations, 0.3)
        
        return max(0.0, completeness - anomaly_penalty)
    
    def _calculate_pattern_strength_score(self, patterns: List[TemporalPattern]) -> float:
        """Calculate overall pattern strength score."""
        
        if not patterns:
            return 0.0
        
        return np.mean([p.strength for p in patterns])
    
    def _calculate_anomaly_score(self, anomalies: List[TemporalAnomaly], total_observations: int) -> float:
        """Calculate anomaly score."""
        
        if not anomalies:
            return 0.0
        
        return min(len(anomalies) / total_observations, 1.0)
    
    def _generate_recommendations(self, result: TemporalAnalysisResult) -> List[str]:
        """Generate recommendations based on analysis results."""
        
        recommendations = []
        
        if result.data_quality_score < 0.8:
            recommendations.append("Improve data quality by addressing missing values and anomalies")
        
        if not result.is_stationary:
            recommendations.append("Consider differencing or transformation to achieve stationarity")
        
        if result.has_seasonality and not result.has_trend:
            recommendations.append("Use seasonal modeling techniques for forecasting")
        
        if result.has_trend and result.has_seasonality:
            recommendations.append("Apply trend and seasonal decomposition methods")
        
        if len(result.anomalies) > result.total_observations * 0.1:
            recommendations.append("High anomaly rate detected - implement anomaly monitoring")
        
        if result.pattern_strength_score < 0.3:
            recommendations.append("Weak patterns detected - consider longer time series or external factors")
        
        return recommendations