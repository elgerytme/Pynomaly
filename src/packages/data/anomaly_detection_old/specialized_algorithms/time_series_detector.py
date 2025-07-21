"""Time series anomaly detection algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import warnings

from simplified_services.core_detection_service import DetectionResult


@dataclass
class TimeSeriesConfig:
    """Configuration for time series anomaly detection."""
    window_size: int = 50
    contamination: float = 0.1
    method: str = "statistical"  # "statistical", "isolation", "lstm", "prophet"
    seasonal_period: Optional[int] = None
    trend_detection: bool = True
    use_differencing: bool = True
    confidence_level: float = 0.95


class TimeSeriesDetector:
    """Time series anomaly detection using multiple approaches.
    
    This class provides specialized anomaly detection for time series data:
    - Statistical methods (Z-score, IQR, seasonal decomposition)
    - Moving window-based detection
    - Trend and seasonality-aware detection
    - Change point detection
    - Simple LSTM-style pattern detection (without deep learning dependencies)
    """

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        """Initialize time series detector.
        
        Args:
            config: Time series detection configuration
        """
        self.config = config or TimeSeriesConfig()

    def detect_anomalies(
        self,
        time_series: npt.NDArray[np.floating],
        timestamps: Optional[npt.NDArray] = None,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in time series data.
        
        Args:
            time_series: 1D time series data
            timestamps: Optional timestamps corresponding to data points
            **kwargs: Additional parameters
            
        Returns:
            DetectionResult with anomaly predictions and scores
        """
        if len(time_series.shape) != 1:
            raise ValueError("Time series must be 1-dimensional")
        
        if self.config.method == "statistical":
            return self._statistical_detection(time_series, timestamps)
        elif self.config.method == "isolation":
            return self._isolation_detection(time_series)
        elif self.config.method == "lstm":
            return self._pattern_detection(time_series)
        elif self.config.method == "prophet":
            return self._trend_seasonal_detection(time_series, timestamps)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _statistical_detection(
        self,
        time_series: npt.NDArray[np.floating],
        timestamps: Optional[npt.NDArray] = None
    ) -> DetectionResult:
        """Statistical anomaly detection for time series."""
        n_points = len(time_series)
        anomalies = np.zeros(n_points, dtype=int)
        scores = np.zeros(n_points, dtype=float)
        
        # Apply differencing if enabled
        if self.config.use_differencing and n_points > 1:
            diff_series = np.diff(time_series)
            diff_series = np.concatenate([[0], diff_series])  # Pad first element
        else:
            diff_series = time_series
        
        # Moving window detection
        for i in range(self.config.window_size, n_points):
            window_start = max(0, i - self.config.window_size)
            window_data = diff_series[window_start:i]
            current_value = diff_series[i]
            
            if len(window_data) < 5:  # Need minimum data
                continue
            
            # Calculate statistics
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std < 1e-8:  # Avoid division by zero
                z_score = 0.0
            else:
                z_score = abs((current_value - mean) / std)
            
            scores[i] = z_score
            
            # Determine threshold based on confidence level
            threshold = stats.norm.ppf((1 + self.config.confidence_level) / 2)
            
            if z_score > threshold:
                anomalies[i] = 1
        
        # Apply contamination-based filtering
        if np.sum(anomalies) > 0:
            anomalies = self._apply_contamination_filtering(scores, anomalies)
        
        return DetectionResult(
            predictions=anomalies,
            scores=scores,
            algorithm=f"timeseries_{self.config.method}",
            contamination=self.config.contamination,
            metadata={
                "method": "statistical",
                "window_size": self.config.window_size,
                "differencing": self.config.use_differencing,
                "confidence_level": self.config.confidence_level
            }
        )

    def _isolation_detection(self, time_series: npt.NDArray[np.floating]) -> DetectionResult:
        """Isolation forest-based detection for time series."""
        # Create features from time series
        features = self._create_time_series_features(time_series)
        
        # Use existing detection service
        from simplified_services.core_detection_service import CoreDetectionService
        detection_service = CoreDetectionService()
        
        result = detection_service.detect_anomalies(
            features,
            algorithm="iforest", 
            contamination=self.config.contamination
        )
        
        result.algorithm = "timeseries_isolation"
        result.metadata.update({
            "method": "isolation_forest",
            "n_features": features.shape[1]
        })
        
        return result

    def _pattern_detection(self, time_series: npt.NDArray[np.floating]) -> DetectionResult:
        """Pattern-based anomaly detection (simplified LSTM-style)."""
        n_points = len(time_series)
        anomalies = np.zeros(n_points, dtype=int)
        scores = np.zeros(n_points, dtype=float)
        
        sequence_length = min(self.config.window_size, 20)
        
        for i in range(sequence_length, n_points):
            # Extract sequence
            sequence = time_series[i-sequence_length:i]
            current_value = time_series[i]
            
            # Simple pattern prediction (moving average with trend)
            recent_trend = np.mean(np.diff(sequence[-5:]))  # Recent trend
            predicted_value = sequence[-1] + recent_trend
            
            # Calculate prediction error
            error = abs(current_value - predicted_value)
            
            # Normalize error by recent variability
            recent_std = np.std(sequence[-10:]) if len(sequence) >= 10 else np.std(sequence)
            if recent_std < 1e-8:
                normalized_error = 0.0
            else:
                normalized_error = error / recent_std
            
            scores[i] = normalized_error
            
            # Threshold based on pattern consistency
            if normalized_error > 2.5:  # Adjustable threshold
                anomalies[i] = 1
        
        # Apply contamination filtering
        if np.sum(anomalies) > 0:
            anomalies = self._apply_contamination_filtering(scores, anomalies)
        
        return DetectionResult(
            predictions=anomalies,
            scores=scores,
            algorithm="timeseries_pattern",
            contamination=self.config.contamination,
            metadata={
                "method": "pattern_detection",
                "sequence_length": sequence_length
            }
        )

    def _trend_seasonal_detection(
        self,
        time_series: npt.NDArray[np.floating],
        timestamps: Optional[npt.NDArray] = None
    ) -> DetectionResult:
        """Trend and seasonal decomposition-based detection."""
        n_points = len(time_series)
        
        # Simple trend estimation using moving averages
        trend = self._estimate_trend(time_series)
        
        # Detrend the series
        detrended = time_series - trend
        
        # Seasonal component estimation
        seasonal = self._estimate_seasonal(detrended)
        
        # Residual component
        residual = detrended - seasonal
        
        # Detect anomalies in residual
        anomalies = np.zeros(n_points, dtype=int)
        scores = np.abs(residual)
        
        # Threshold based on residual statistics
        residual_std = np.std(residual)
        threshold = residual_std * 2.5  # Adjustable threshold
        
        anomalies[scores > threshold] = 1
        
        # Apply contamination filtering
        if np.sum(anomalies) > 0:
            anomalies = self._apply_contamination_filtering(scores, anomalies)
        
        return DetectionResult(
            predictions=anomalies,
            scores=scores,
            algorithm="timeseries_seasonal",
            contamination=self.config.contamination,
            metadata={
                "method": "trend_seasonal",
                "seasonal_period": self.config.seasonal_period,
                "trend_detected": np.std(trend) > np.std(time_series) * 0.1
            }
        )

    def _create_time_series_features(self, time_series: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Create features from time series for traditional ML algorithms."""
        n_points = len(time_series)
        window_size = min(self.config.window_size, n_points // 4)
        
        features = []
        
        for i in range(window_size, n_points):
            window = time_series[i-window_size:i]
            
            # Statistical features
            feature_vector = [
                np.mean(window),           # Mean
                np.std(window),            # Standard deviation
                np.min(window),            # Minimum
                np.max(window),            # Maximum
                np.median(window),         # Median
                time_series[i],            # Current value
            ]
            
            # Trend features
            if len(window) > 1:
                trend = np.polyfit(range(len(window)), window, 1)[0]
                feature_vector.append(trend)
            else:
                feature_vector.append(0.0)
            
            # Difference features
            if i > 0:
                feature_vector.append(time_series[i] - time_series[i-1])  # First difference
            else:
                feature_vector.append(0.0)
            
            features.append(feature_vector)
        
        # Pad beginning with repeated first feature
        if features:
            first_features = [features[0]] * window_size
            features = first_features + features
        
        return np.array(features)

    def _estimate_trend(self, time_series: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Estimate trend component using moving averages."""
        trend_window = min(len(time_series) // 4, 50)
        trend = np.zeros_like(time_series)
        
        for i in range(len(time_series)):
            start_idx = max(0, i - trend_window // 2)
            end_idx = min(len(time_series), i + trend_window // 2 + 1)
            trend[i] = np.mean(time_series[start_idx:end_idx])
        
        return trend

    def _estimate_seasonal(self, detrended_series: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Estimate seasonal component."""
        if self.config.seasonal_period is None:
            # Try to detect seasonality
            seasonal_period = self._detect_seasonality(detrended_series)
        else:
            seasonal_period = self.config.seasonal_period
        
        if seasonal_period is None or seasonal_period >= len(detrended_series) // 2:
            return np.zeros_like(detrended_series)
        
        # Simple seasonal estimation
        seasonal = np.zeros_like(detrended_series)
        
        for i in range(len(detrended_series)):
            seasonal_indices = list(range(i % seasonal_period, len(detrended_series), seasonal_period))
            if len(seasonal_indices) > 1:
                seasonal[i] = np.mean(detrended_series[seasonal_indices])
        
        return seasonal

    def _detect_seasonality(self, time_series: npt.NDArray[np.floating]) -> Optional[int]:
        """Simple seasonality detection using autocorrelation."""
        if len(time_series) < 20:
            return None
        
        max_period = min(len(time_series) // 3, 100)
        
        best_period = None
        best_correlation = 0.0
        
        for period in range(2, max_period):
            if len(time_series) < 2 * period:
                continue
            
            # Calculate autocorrelation at this lag
            correlation = np.corrcoef(
                time_series[:-period], 
                time_series[period:]
            )[0, 1]
            
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_period = period
        
        # Only return period if correlation is significant
        return best_period if best_correlation > 0.3 else None

    def _apply_contamination_filtering(
        self,
        scores: npt.NDArray[np.floating],
        anomalies: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]:
        """Apply contamination-based filtering to anomaly predictions."""
        n_points = len(anomalies)
        target_anomalies = max(1, int(self.config.contamination * n_points))
        
        if np.sum(anomalies) <= target_anomalies:
            return anomalies
        
        # Keep only top scoring anomalies
        anomaly_indices = np.where(anomalies == 1)[0]
        anomaly_scores = scores[anomaly_indices]
        
        # Sort by score (descending)
        sorted_indices = anomaly_indices[np.argsort(anomaly_scores)[::-1]]
        
        # Keep only top N
        filtered_anomalies = np.zeros_like(anomalies)
        filtered_anomalies[sorted_indices[:target_anomalies]] = 1
        
        return filtered_anomalies

    def detect_change_points(
        self,
        time_series: npt.NDArray[np.floating],
        min_segment_length: int = 10
    ) -> List[int]:
        """Detect change points in time series.
        
        Args:
            time_series: Input time series
            min_segment_length: Minimum length of segments
            
        Returns:
            List of change point indices
        """
        change_points = []
        
        if len(time_series) < 2 * min_segment_length:
            return change_points
        
        # Simple change point detection using moving statistics
        window_size = min(min_segment_length, len(time_series) // 10)
        
        for i in range(window_size, len(time_series) - window_size):
            # Compare statistics before and after potential change point
            before = time_series[i-window_size:i]
            after = time_series[i:i+window_size]
            
            # Statistical test for difference in means
            if len(before) > 1 and len(after) > 1:
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if not np.isnan(p_value) and p_value < 0.01:  # Significant change
                    change_points.append(i)
        
        # Remove close change points
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] >= min_segment_length:
                filtered_change_points.append(cp)
        
        return filtered_change_points

    def get_decomposition(
        self,
        time_series: npt.NDArray[np.floating]
    ) -> Dict[str, npt.NDArray[np.floating]]:
        """Get time series decomposition into trend, seasonal, and residual components.
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        trend = self._estimate_trend(time_series)
        detrended = time_series - trend
        seasonal = self._estimate_seasonal(detrended)
        residual = detrended - seasonal
        
        return {
            "original": time_series,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual
        }