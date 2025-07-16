"""Time series anomaly detection adapter using statistical and machine learning methods.

This adapter provides time series-specific anomaly detection algorithms that work
with temporal data, including seasonality detection, trend analysis, and change point detection.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from monorepo.domain.entities import Dataset, DetectionResult, Detector
from monorepo.domain.exceptions import AdapterError, AlgorithmNotFoundError
from monorepo.domain.value_objects import AnomalyScore
from monorepo.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class StatisticalTimeSeriesDetector:
    """Statistical methods for time series anomaly detection."""

    def __init__(self, window_size: int = 10, threshold_factor: float = 3.0):
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.stats_cache = {}

    def fit(self, data: np.ndarray) -> None:
        """Fit the statistical detector."""
        # Calculate rolling statistics
        df = pd.DataFrame({"value": data})

        # Rolling mean and std
        self.stats_cache["rolling_mean"] = (
            df["value"]
            .rolling(window=self.window_size)
            .mean()
            .fillna(df["value"].mean())
        )
        self.stats_cache["rolling_std"] = (
            df["value"].rolling(window=self.window_size).std().fillna(df["value"].std())
        )

        # Global statistics
        self.stats_cache["global_mean"] = np.mean(data)
        self.stats_cache["global_std"] = np.std(data)

        # Percentiles for threshold calculation
        self.stats_cache["p95"] = np.percentile(data, 95)
        self.stats_cache["p05"] = np.percentile(data, 5)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using statistical methods."""
        scores = []

        for i, value in enumerate(data):
            # Z-score based on rolling statistics
            if i < len(self.stats_cache["rolling_mean"]):
                local_mean = self.stats_cache["rolling_mean"].iloc[i]
                local_std = self.stats_cache["rolling_std"].iloc[i]
            else:
                local_mean = self.stats_cache["global_mean"]
                local_std = self.stats_cache["global_std"]

            if local_std == 0:
                z_score = 0
            else:
                z_score = abs(value - local_mean) / local_std

            # Percentile-based score
            if value > self.stats_cache["p95"]:
                percentile_score = (value - self.stats_cache["p95"]) / (
                    self.stats_cache["p95"] - self.stats_cache["global_mean"]
                )
            elif value < self.stats_cache["p05"]:
                percentile_score = (self.stats_cache["p05"] - value) / (
                    self.stats_cache["global_mean"] - self.stats_cache["p05"]
                )
            else:
                percentile_score = 0

            # Combined score
            combined_score = max(z_score, percentile_score)
            scores.append(combined_score)

        return np.array(scores)


class SeasonalDecompositionDetector:
    """Seasonal decomposition-based anomaly detection."""

    def __init__(self, period: int | None = None, model: str = "additive"):
        self.period = period
        self.model = model
        self.baseline_residuals = None
        self.seasonal_component = None
        self.trend_component = None

    def _simple_seasonal_decompose(
        self, data: np.ndarray, period: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple seasonal decomposition implementation."""
        n = len(data)

        # Calculate seasonal component using simple moving average
        seasonal = np.zeros(n)
        for i in range(period):
            seasonal_values = data[i::period]
            seasonal[i::period] = np.mean(seasonal_values)

        # Calculate trend using centered moving average
        if n >= 2 * period:
            trend = np.convolve(data, np.ones(period) / period, mode="same")
        else:
            trend = np.full(n, np.mean(data))

        # Calculate residuals
        if self.model == "additive":
            residuals = data - trend - seasonal
        else:  # multiplicative
            residuals = data / (trend + 1e-8) / (seasonal + 1e-8)

        return trend, seasonal, residuals

    def fit(self, data: np.ndarray) -> None:
        """Fit the seasonal decomposition detector."""
        if self.period is None:
            # Auto-detect period using autocorrelation
            self.period = self._detect_period(data)

        if self.period > 1 and len(data) >= 2 * self.period:
            (
                self.trend_component,
                self.seasonal_component,
                self.baseline_residuals,
            ) = self._simple_seasonal_decompose(data, self.period)
        else:
            # Fall back to simple statistics
            self.trend_component = np.full(len(data), np.mean(data))
            self.seasonal_component = np.zeros(len(data))
            self.baseline_residuals = data - self.trend_component

        # Calculate residual statistics
        self.residual_mean = np.mean(self.baseline_residuals)
        self.residual_std = np.std(self.baseline_residuals)

    def _detect_period(self, data: np.ndarray) -> int:
        """Detect period using autocorrelation."""
        if len(data) < 10:
            return 1

        # Calculate autocorrelation
        max_lag = min(len(data) // 4, 50)
        autocorr = []

        for lag in range(1, max_lag + 1):
            if lag >= len(data):
                break
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append(corr)
            else:
                autocorr.append(0)

        if not autocorr:
            return 1

        # Find peaks in autocorrelation
        autocorr = np.array(autocorr)
        peaks, _ = find_peaks(autocorr, height=0.3, distance=3)

        if len(peaks) > 0:
            return peaks[0] + 1  # Add 1 because lag starts from 1
        else:
            return 1

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using seasonal decomposition."""
        if len(data) != len(self.baseline_residuals):
            # For new data, we need to recompute decomposition
            if self.period > 1 and len(data) >= 2 * self.period:
                trend, seasonal, residuals = self._simple_seasonal_decompose(
                    data, self.period
                )
            else:
                trend = np.full(len(data), np.mean(data))
                np.zeros(len(data))
                residuals = data - trend
        else:
            residuals = self.baseline_residuals

        # Calculate anomaly scores based on residual deviation
        if self.residual_std == 0:
            scores = np.zeros(len(residuals))
        else:
            scores = np.abs(residuals - self.residual_mean) / self.residual_std

        return scores


class ChangePointDetector:
    """Change point detection for time series anomalies."""

    def __init__(self, window_size: int = 10, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.baseline_stats = None

    def fit(self, data: np.ndarray) -> None:
        """Fit the change point detector."""
        # Calculate baseline statistics for each window
        self.baseline_stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
        }

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Detect change points as anomalies."""
        scores = np.zeros(len(data))

        for i in range(self.window_size, len(data)):
            # Current window
            current_window = data[i - self.window_size : i]

            # Previous window
            if i >= 2 * self.window_size:
                prev_window = data[i - 2 * self.window_size : i - self.window_size]
            else:
                prev_window = (
                    data[: self.window_size]
                    if len(data) >= self.window_size
                    else data[:i]
                )

            if len(prev_window) == 0:
                continue

            # Statistical tests for change point
            try:
                # Mean change test
                mean_change = abs(np.mean(current_window) - np.mean(prev_window))
                mean_score = mean_change / (self.baseline_stats["std"] + 1e-8)

                # Variance change test
                var_change = abs(np.var(current_window) - np.var(prev_window))
                var_score = var_change / (self.baseline_stats["std"] ** 2 + 1e-8)

                # Combined score
                scores[i] = max(mean_score, var_score)

            except (ZeroDivisionError, ValueError):
                scores[i] = 0

        return scores * self.sensitivity


class TimeSeriesAdapter(DetectorProtocol):
    """Time series anomaly detection adapter supporting multiple algorithms."""

    _algorithm_map = {
        "StatisticalTS": StatisticalTimeSeriesDetector,
        "SeasonalDecomposition": SeasonalDecompositionDetector,
        "ChangePointDetection": ChangePointDetector,
    }

    def __init__(self, detector: Detector):
        """Initialize time series adapter.

        Args:
            detector: Detector entity with algorithm configuration
        """
        self.detector = detector
        self._model = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        """Initialize the time series algorithm."""
        if self.detector.algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm_name}' not found. "
                f"Available time series algorithms: {available}"
            )

        algorithm_class = self._algorithm_map[self.detector.algorithm_name]
        params = self.detector.parameters.copy()

        # Create algorithm instance with parameters
        try:
            if self.detector.algorithm_name == "StatisticalTS":
                self._model = algorithm_class(
                    window_size=params.get("window_size", 10),
                    threshold_factor=params.get("threshold_factor", 3.0),
                )
            elif self.detector.algorithm_name == "SeasonalDecomposition":
                self._model = algorithm_class(
                    period=params.get("period", None),
                    model=params.get("model", "additive"),
                )
            elif self.detector.algorithm_name == "ChangePointDetection":
                self._model = algorithm_class(
                    window_size=params.get("window_size", 10),
                    sensitivity=params.get("sensitivity", 2.0),
                )
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize {self.detector.algorithm_name}: {e}"
            )

    def fit(self, dataset: Dataset) -> None:
        """Fit the time series detector.

        Args:
            dataset: Training dataset with time series data
        """
        try:
            # Prepare time series data
            time_series_data = self._prepare_time_series_data(dataset)

            # Fit the model
            self._model.fit(time_series_data)

            self.detector.is_fitted = True
            self._is_fitted = True
            logger.info(
                f"Successfully fitted time series detector: {self.detector.algorithm_name}"
            )

        except Exception as e:
            raise AdapterError(f"Failed to fit time series model: {e}")

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies in time series data.

        Args:
            dataset: Dataset with time series to analyze

        Returns:
            Detection results with anomaly scores and labels
        """
        if not self._is_fitted or self._model is None:
            raise AdapterError("Model must be fitted before prediction")

        try:
            # Prepare data
            time_series_data = self._prepare_time_series_data(dataset)

            # Get anomaly scores
            scores = self._model.predict(time_series_data)

            # Normalize scores to [0, 1]
            if np.max(scores) > np.min(scores):
                normalized_scores = (scores - np.min(scores)) / (
                    np.max(scores) - np.min(scores)
                )
            else:
                normalized_scores = np.zeros_like(scores)

            # Calculate threshold and labels
            contamination = self.detector.parameters.get("contamination", 0.1)
            threshold = np.percentile(normalized_scores, (1 - contamination) * 100)
            labels = (normalized_scores > threshold).astype(int)

            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(value=float(score), method=self.detector.algorithm_name)
                for score in normalized_scores
            ]

            # Create anomaly objects for detected anomalies
            from monorepo.domain.entities.anomaly import Anomaly

            anomalies = []
            anomaly_indices = np.where(labels == 1)[0]

            for idx in anomaly_indices:
                anomaly = Anomaly(
                    score=anomaly_scores[idx],
                    data_point={
                        "index": int(idx),
                        "value": float(time_series_data[idx]),
                    },
                    detector_name=self.detector.algorithm_name,
                    metadata={"time_series_index": int(idx)},
                )
                anomalies.append(anomaly)

            return DetectionResult(
                detector_id=self.detector.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels,
                threshold=float(threshold),
                metadata={
                    "algorithm": self.detector.algorithm_name,
                    "n_anomalies": int(np.sum(labels)),
                    "contamination_rate": float(np.sum(labels) / len(labels)),
                    "model_type": "time_series",
                    "data_length": len(time_series_data),
                    "detected_period": (
                        getattr(self._model, "period", None)
                        if hasattr(self._model, "period")
                        else None
                    ),
                },
            )

        except Exception as e:
            raise AdapterError(f"Failed to predict with time series model: {e}")

    def _prepare_time_series_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare time series data for analysis.

        Args:
            dataset: Input dataset

        Returns:
            1D numpy array representing the time series
        """
        df = dataset.data

        # Handle different time series formats
        if dataset.target_column and dataset.target_column in df.columns:
            # Use specified target column as time series
            time_series = df[dataset.target_column].values
        else:
            # Try to find numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) == 1:
                # Single numeric column - use as time series
                time_series = df[numeric_cols[0]].values
            elif len(numeric_cols) > 1:
                # Multiple columns - use first one or try to detect main series
                if "value" in numeric_cols:
                    time_series = df["value"].values
                elif "y" in numeric_cols:
                    time_series = df["y"].values
                else:
                    time_series = df[numeric_cols[0]].values
            else:
                raise AdapterError("No numeric columns found for time series analysis")

        # Handle missing values
        if np.any(np.isnan(time_series)):
            # Simple forward fill + backward fill
            time_series_clean = pd.Series(time_series).ffill().bfill().values
        else:
            time_series_clean = time_series

        if len(time_series_clean) == 0:
            raise AdapterError("Time series data is empty")

        return time_series_clean

    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence score.

        Args:
            score: Anomaly score
            threshold: Detection threshold

        Returns:
            Confidence value between 0 and 1
        """
        if score <= threshold:
            return 1.0 - (score / threshold) * 0.5
        else:
            return 0.5 + min((score - threshold) / threshold * 0.5, 0.5)

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported time series algorithms.

        Returns:
            List of algorithm names
        """
        return list(cls._algorithm_map.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Algorithm metadata and parameters
        """
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")

        info = {
            "StatisticalTS": {
                "name": "Statistical Time Series",
                "type": "Statistical",
                "description": "Statistical anomaly detection using rolling statistics and percentiles",
                "parameters": {
                    "window_size": {
                        "type": "int",
                        "default": 10,
                        "description": "Rolling window size",
                    },
                    "threshold_factor": {
                        "type": "float",
                        "default": 3.0,
                        "description": "Z-score threshold multiplier",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "univariate_time_series",
                    "real_time_detection",
                    "simple_patterns",
                ],
                "pros": [
                    "Fast computation",
                    "No training required",
                    "Interpretable results",
                ],
                "cons": ["May miss complex patterns", "Sensitive to parameter choices"],
            },
            "SeasonalDecomposition": {
                "name": "Seasonal Decomposition",
                "type": "Statistical",
                "description": "Anomaly detection based on seasonal decomposition of time series",
                "parameters": {
                    "period": {
                        "type": "int",
                        "default": "auto",
                        "description": "Seasonal period (auto-detected if None)",
                    },
                    "model": {
                        "type": "str",
                        "default": "additive",
                        "description": "Decomposition model (additive/multiplicative)",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "seasonal_data",
                    "trend_analysis",
                    "long_term_patterns",
                ],
                "pros": [
                    "Handles seasonality",
                    "Separates trend from noise",
                    "Good for business data",
                ],
                "cons": ["Requires sufficient data", "Assumes stable seasonality"],
            },
            "ChangePointDetection": {
                "name": "Change Point Detection",
                "type": "Statistical",
                "description": "Detects abrupt changes in time series statistical properties",
                "parameters": {
                    "window_size": {
                        "type": "int",
                        "default": 10,
                        "description": "Comparison window size",
                    },
                    "sensitivity": {
                        "type": "float",
                        "default": 2.0,
                        "description": "Detection sensitivity",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "regime_changes",
                    "structural_breaks",
                    "monitoring_systems",
                ],
                "pros": [
                    "Detects structural changes",
                    "Fast detection",
                    "Good for monitoring",
                ],
                "cons": ["May miss gradual changes", "Sensitive to noise"],
            },
        }

        return info.get(algorithm, {})
