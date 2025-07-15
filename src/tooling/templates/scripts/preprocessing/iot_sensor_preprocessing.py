#!/usr/bin/env python3
"""
IoT Sensor Data Preprocessing Pipeline Template

This template provides a comprehensive preprocessing pipeline specifically designed
for IoT sensor data including time series processing, sensor fusion, and anomaly-ready preparation.
"""

import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Time series processing
import logging

# Statistical imports
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer

# Data processing imports
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTSensorPreprocessor:
    """
    Comprehensive preprocessing pipeline for IoT sensor datasets.

    Features:
    - Time series preprocessing and resampling
    - Sensor data cleaning and calibration
    - Multi-sensor fusion and alignment
    - Environmental data normalization
    - Anomaly detection preparation
    - Real-time processing optimization
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        preserve_original: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the IoT sensor data preprocessor.

        Args:
            config: Configuration dictionary for preprocessing steps
            preserve_original: Whether to preserve original column values
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.preserve_original = preserve_original
        self.verbose = verbose

        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}

        # IoT-specific components
        self.sensor_calibrations = {}
        self.temporal_features = {}
        self.fusion_weights = {}

        # Metadata tracking
        self.preprocessing_steps = []
        self.data_profile = {}
        self.sensor_metadata = {}

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for IoT sensor data preprocessing."""
        return {
            "temporal": {
                "resample_frequency": None,  # '1min', '5min', '1H'
                "interpolation_method": "linear",  # 'linear', 'cubic', 'nearest'
                "seasonal_decomposition": True,
                "stationarity_test": True,
            },
            "missing_values": {
                "strategy": "interpolation",  # 'interpolation', 'forward_fill', 'knn'
                "max_gap_minutes": 30,  # Maximum gap to interpolate
                "threshold": 0.3,  # Drop sensors with >30% missing
            },
            "outliers": {
                "method": "iqr_per_sensor",  # 'iqr_per_sensor', 'zscore', 'isolation_forest'
                "threshold": 2.5,
                "action": "interpolate",  # 'remove', 'cap', 'interpolate'
                "window_size": 100,  # For rolling statistics
            },
            "sensor_fusion": {
                "enable": True,
                "correlation_threshold": 0.8,
                "redundancy_removal": True,
                "weighted_averaging": True,
            },
            "scaling": {
                "method": "robust",  # 'standard', 'minmax', 'robust'
                "per_sensor": True,  # Scale each sensor independently
                "feature_range": (0, 1),
            },
            "feature_engineering": {
                "rolling_statistics": True,
                "window_sizes": [5, 10, 30],  # Minutes
                "lag_features": True,
                "max_lags": 5,
                "seasonal_features": True,
                "sensor_interactions": True,
            },
            "quality_checks": {
                "range_validation": True,
                "drift_detection": True,
                "calibration_check": True,
                "noise_analysis": True,
            },
        }

    def preprocess(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        sensor_columns: list[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply comprehensive preprocessing pipeline to IoT sensor data.

        Args:
            data: Input DataFrame with timestamp and sensor readings
            timestamp_column: Name of timestamp column
            sensor_columns: List of sensor column names (auto-detect if None)

        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        logger.info("Starting IoT sensor data preprocessing pipeline")

        # Create copy to avoid modifying original
        df = data.copy()
        original_shape = df.shape

        # Auto-detect sensor columns if not provided
        if sensor_columns is None:
            sensor_columns = [
                col
                for col in df.columns
                if col != timestamp_column and df[col].dtype in ["int64", "float64"]
            ]

        # 1. Temporal Data Preparation
        self._log_step("Temporal Data Preparation")
        df = self._prepare_temporal_data(df, timestamp_column)

        # 2. Sensor Data Quality Assessment
        self._log_step("Sensor Data Quality Assessment")
        quality_results = self._assess_sensor_quality(df, sensor_columns)

        # 3. Handle Missing Values
        self._log_step("Missing Value Treatment")
        df = self._handle_missing_values(df, sensor_columns, timestamp_column)

        # 4. Outlier Detection and Treatment
        self._log_step("Outlier Detection and Treatment")
        df = self._handle_outliers(df, sensor_columns)

        # 5. Sensor Calibration and Normalization
        self._log_step("Sensor Calibration and Normalization")
        df = self._calibrate_sensors(df, sensor_columns)

        # 6. Feature Engineering
        self._log_step("IoT Feature Engineering")
        df = self._engineer_iot_features(df, sensor_columns, timestamp_column)

        # 7. Sensor Fusion
        self._log_step("Sensor Fusion")
        df = self._fuse_sensors(df, sensor_columns)

        # 8. Feature Scaling
        self._log_step("Feature Scaling")
        df = self._scale_features(df, sensor_columns)

        # 9. Feature Selection
        self._log_step("Feature Selection")
        df = self._select_features(df)

        # 10. Final Validation
        self._log_step("Final Validation")
        final_validation = self._final_validation(df, original_shape)

        # Prepare metadata
        metadata = {
            "preprocessing_steps": self.preprocessing_steps,
            "data_profile": self.data_profile,
            "sensor_metadata": self.sensor_metadata,
            "quality_results": quality_results,
            "final_validation": final_validation,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "config": self.config,
            "sensor_columns": sensor_columns,
        }

        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        return df, metadata

    def _prepare_temporal_data(
        self, df: pd.DataFrame, timestamp_column: str
    ) -> pd.DataFrame:
        """Prepare temporal aspects of the data."""
        # Convert timestamp to datetime
        if df[timestamp_column].dtype == "object":
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Set as index for time series operations
        df = df.set_index(timestamp_column)
        df = df.sort_index()

        # Resample if configured
        resample_freq = self.config["temporal"]["resample_frequency"]
        if resample_freq:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns].resample(resample_freq).mean()

            self.preprocessing_steps.append(
                {
                    "step": "temporal_resampling",
                    "frequency": resample_freq,
                    "method": "mean",
                }
            )

        return df

    def _assess_sensor_quality(
        self, df: pd.DataFrame, sensor_columns: list[str]
    ) -> dict[str, Any]:
        """Assess quality of sensor data."""
        quality_results = {
            "total_sensors": len(sensor_columns),
            "time_range": (df.index.min(), df.index.max()),
            "sampling_rate_seconds": (
                (df.index[1] - df.index[0]).total_seconds() if len(df) > 1 else None
            ),
            "sensor_analysis": {},
        }

        for sensor in sensor_columns:
            if sensor in df.columns:
                sensor_data = df[sensor]

                # Basic statistics
                sensor_stats = {
                    "missing_ratio": sensor_data.isnull().sum() / len(sensor_data),
                    "mean": sensor_data.mean(),
                    "std": sensor_data.std(),
                    "min": sensor_data.min(),
                    "max": sensor_data.max(),
                    "range": sensor_data.max() - sensor_data.min(),
                    "outlier_ratio": None,
                    "noise_level": None,
                    "drift_detected": False,
                }

                # Outlier analysis using IQR
                Q1 = sensor_data.quantile(0.25)
                Q3 = sensor_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (sensor_data < (Q1 - 1.5 * IQR)) | (sensor_data > (Q3 + 1.5 * IQR))
                ).sum()
                sensor_stats["outlier_ratio"] = outliers / len(sensor_data)

                # Noise analysis (using signal-to-noise ratio approximation)
                if sensor_stats["std"] > 0:
                    sensor_stats["noise_level"] = sensor_stats["std"] / abs(
                        sensor_stats["mean"]
                    )

                # Simple drift detection (compare first and last quarters)
                if len(sensor_data) > 100:
                    quarter_size = len(sensor_data) // 4
                    first_quarter_mean = sensor_data.iloc[:quarter_size].mean()
                    last_quarter_mean = sensor_data.iloc[-quarter_size:].mean()
                    drift_threshold = 2 * sensor_stats["std"]
                    sensor_stats["drift_detected"] = (
                        abs(last_quarter_mean - first_quarter_mean) > drift_threshold
                    )

                quality_results["sensor_analysis"][sensor] = sensor_stats

        self.data_profile["quality"] = quality_results
        return quality_results

    def _handle_missing_values(
        self, df: pd.DataFrame, sensor_columns: list[str], timestamp_column: str
    ) -> pd.DataFrame:
        """Handle missing values in sensor data using time-aware methods."""
        strategy = self.config["missing_values"]["strategy"]
        max_gap_minutes = self.config["missing_values"]["max_gap_minutes"]
        threshold = self.config["missing_values"]["threshold"]

        # Remove sensors with too many missing values
        missing_ratios = df[sensor_columns].isnull().sum() / len(df)
        sensors_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()

        if sensors_to_drop:
            df = df.drop(columns=sensors_to_drop)
            sensor_columns = [
                col for col in sensor_columns if col not in sensors_to_drop
            ]

            self.preprocessing_steps.append(
                {
                    "step": "drop_high_missing_sensors",
                    "sensors_dropped": sensors_to_drop,
                    "threshold": threshold,
                }
            )

        # Handle remaining missing values
        interpolation_info = {}

        for sensor in sensor_columns:
            if sensor in df.columns:
                sensor_data = df[sensor]
                missing_mask = sensor_data.isnull()

                if missing_mask.any():
                    if strategy == "interpolation":
                        # Time-aware interpolation
                        method = self.config["missing_values"]["interpolation_method"]

                        # Only interpolate gaps smaller than max_gap_minutes
                        gaps = (
                            missing_mask.astype(int)
                            .groupby((~missing_mask).cumsum())
                            .cumsum()
                        )
                        max_gap_samples = max_gap_minutes  # Assuming 1-minute sampling

                        # Create mask for interpolatable gaps
                        interpolate_mask = gaps <= max_gap_samples

                        # Interpolate allowed gaps
                        if method == "linear":
                            df[sensor] = sensor_data.interpolate(method="linear")
                        elif method == "cubic":
                            df[sensor] = sensor_data.interpolate(method="cubic")
                        elif method == "nearest":
                            df[sensor] = sensor_data.interpolate(method="nearest")

                        # Set large gaps back to NaN
                        df.loc[missing_mask & ~interpolate_mask, sensor] = np.nan

                        interpolation_info[sensor] = {
                            "method": method,
                            "interpolated_points": (
                                missing_mask & interpolate_mask
                            ).sum(),
                            "remaining_missing": df[sensor].isnull().sum(),
                        }

                    elif strategy == "forward_fill":
                        df[sensor] = sensor_data.fillna(method="ffill")
                        interpolation_info[sensor] = "forward_fill"

                    elif strategy == "knn":
                        # Use KNN imputer for multivariate imputation
                        knn_columns = [
                            col for col in sensor_columns if col in df.columns
                        ]
                        if len(knn_columns) > 1:
                            imputer = KNNImputer(n_neighbors=5)
                            df[knn_columns] = imputer.fit_transform(df[knn_columns])
                            self.imputers["sensors"] = imputer
                            interpolation_info[sensor] = "knn_imputation"

        self.preprocessing_steps.append(
            {
                "step": "missing_value_interpolation",
                "strategy": strategy,
                "interpolation_details": interpolation_info,
            }
        )

        return df

    def _handle_outliers(
        self, df: pd.DataFrame, sensor_columns: list[str]
    ) -> pd.DataFrame:
        """Handle outliers in sensor data using sensor-specific methods."""
        method = self.config["outliers"]["method"]
        threshold = self.config["outliers"]["threshold"]
        action = self.config["outliers"]["action"]
        window_size = self.config["outliers"]["window_size"]

        outlier_info = {}

        for sensor in sensor_columns:
            if sensor in df.columns:
                sensor_data = df[sensor]

                if method == "iqr_per_sensor":
                    # Rolling IQR for adaptive outlier detection
                    rolling_q25 = sensor_data.rolling(
                        window=window_size, center=True
                    ).quantile(0.25)
                    rolling_q75 = sensor_data.rolling(
                        window=window_size, center=True
                    ).quantile(0.75)
                    rolling_iqr = rolling_q75 - rolling_q25

                    lower_bound = rolling_q25 - threshold * rolling_iqr
                    upper_bound = rolling_q75 + threshold * rolling_iqr

                    outliers = (sensor_data < lower_bound) | (sensor_data > upper_bound)

                elif method == "zscore":
                    # Rolling z-score
                    rolling_mean = sensor_data.rolling(
                        window=window_size, center=True
                    ).mean()
                    rolling_std = sensor_data.rolling(
                        window=window_size, center=True
                    ).std()
                    z_scores = np.abs((sensor_data - rolling_mean) / rolling_std)
                    outliers = z_scores > threshold

                outlier_count = outliers.sum()
                if outlier_count > 0:
                    outlier_info[sensor] = outlier_count

                    if action == "interpolate":
                        # Mark outliers as missing and interpolate
                        df.loc[outliers, sensor] = np.nan
                        df[sensor] = df[sensor].interpolate(method="linear")
                    elif action == "cap":
                        if method == "iqr_per_sensor":
                            df.loc[sensor_data < lower_bound, sensor] = lower_bound
                            df.loc[sensor_data > upper_bound, sensor] = upper_bound
                    elif action == "remove":
                        df = df[~outliers]

        self.preprocessing_steps.append(
            {
                "step": "outlier_treatment",
                "method": method,
                "action": action,
                "outliers_found": outlier_info,
            }
        )

        return df

    def _calibrate_sensors(
        self, df: pd.DataFrame, sensor_columns: list[str]
    ) -> pd.DataFrame:
        """Apply sensor calibration and normalization."""
        calibration_info = {}

        for sensor in sensor_columns:
            if sensor in df.columns:
                sensor_data = df[sensor]

                # Remove sensor bias (zero-offset correction)
                if self.config["quality_checks"]["calibration_check"]:
                    # Simple bias correction using the first stable period
                    stable_period = sensor_data.iloc[: min(100, len(sensor_data))]
                    if len(stable_period) > 10:
                        bias = stable_period.mean()

                        # Only apply bias correction if bias is significant
                        if abs(bias) > 0.1 * sensor_data.std():
                            df[sensor] = sensor_data - bias
                            calibration_info[sensor] = f"bias_correction_{bias:.4f}"

                # Noise reduction using moving average
                if self.config["quality_checks"]["noise_analysis"]:
                    noise_level = (
                        sensor_data.std() / abs(sensor_data.mean())
                        if sensor_data.mean() != 0
                        else 0
                    )
                    if noise_level > 0.1:  # High noise threshold
                        # Apply light smoothing
                        smoothed = sensor_data.rolling(window=3, center=True).mean()
                        df[sensor] = smoothed.fillna(sensor_data)
                        calibration_info[sensor] = (
                            calibration_info.get(sensor, "") + "_noise_reduction"
                        )

        if calibration_info:
            self.preprocessing_steps.append(
                {"step": "sensor_calibration", "calibrations_applied": calibration_info}
            )

        return df

    def _engineer_iot_features(
        self, df: pd.DataFrame, sensor_columns: list[str], timestamp_column: str = None
    ) -> pd.DataFrame:
        """Engineer IoT-specific features."""
        new_features = []

        # Time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        new_features.extend(["hour", "day_of_week", "month", "is_weekend"])

        # Rolling statistics
        if self.config["feature_engineering"]["rolling_statistics"]:
            window_sizes = self.config["feature_engineering"]["window_sizes"]

            for sensor in sensor_columns:
                if sensor in df.columns:
                    for window in window_sizes:
                        # Rolling mean, std, min, max
                        df[f"{sensor}_rolling_mean_{window}"] = (
                            df[sensor].rolling(window=window).mean()
                        )
                        df[f"{sensor}_rolling_std_{window}"] = (
                            df[sensor].rolling(window=window).std()
                        )
                        df[f"{sensor}_rolling_min_{window}"] = (
                            df[sensor].rolling(window=window).min()
                        )
                        df[f"{sensor}_rolling_max_{window}"] = (
                            df[sensor].rolling(window=window).max()
                        )

                        new_features.extend(
                            [
                                f"{sensor}_rolling_mean_{window}",
                                f"{sensor}_rolling_std_{window}",
                                f"{sensor}_rolling_min_{window}",
                                f"{sensor}_rolling_max_{window}",
                            ]
                        )

        # Lag features
        if self.config["feature_engineering"]["lag_features"]:
            max_lags = self.config["feature_engineering"]["max_lags"]

            for sensor in sensor_columns:
                if sensor in df.columns:
                    for lag in range(1, max_lags + 1):
                        df[f"{sensor}_lag_{lag}"] = df[sensor].shift(lag)
                        new_features.append(f"{sensor}_lag_{lag}")

        # Sensor interactions
        if (
            self.config["feature_engineering"]["sensor_interactions"]
            and len(sensor_columns) > 1
        ):
            for i, sensor1 in enumerate(sensor_columns):
                for sensor2 in sensor_columns[i + 1 :]:
                    if sensor1 in df.columns and sensor2 in df.columns:
                        # Ratio features
                        df[f"{sensor1}_{sensor2}_ratio"] = df[sensor1] / (
                            df[sensor2] + 1e-8
                        )

                        # Difference features
                        df[f"{sensor1}_{sensor2}_diff"] = df[sensor1] - df[sensor2]

                        new_features.extend(
                            [f"{sensor1}_{sensor2}_ratio", f"{sensor1}_{sensor2}_diff"]
                        )

        # Seasonal decomposition features
        if self.config["feature_engineering"]["seasonal_features"]:
            for sensor in sensor_columns[
                :3
            ]:  # Limit to first 3 sensors for performance
                if sensor in df.columns and len(df) > 100:
                    try:
                        # Simple seasonal decomposition
                        decomposition = seasonal_decompose(
                            df[sensor].dropna(),
                            model="additive",
                            period=min(24, len(df) // 4),  # Daily or available
                        )

                        df[f"{sensor}_trend"] = decomposition.trend
                        df[f"{sensor}_seasonal"] = decomposition.seasonal
                        df[f"{sensor}_residual"] = decomposition.resid

                        new_features.extend(
                            [
                                f"{sensor}_trend",
                                f"{sensor}_seasonal",
                                f"{sensor}_residual",
                            ]
                        )
                    except:
                        # Skip if decomposition fails
                        pass

        self.preprocessing_steps.append(
            {
                "step": "iot_feature_engineering",
                "new_features_count": len(new_features),
                "feature_types": [
                    "temporal",
                    "rolling_stats",
                    "lags",
                    "interactions",
                    "seasonal",
                ],
            }
        )

        return df

    def _fuse_sensors(
        self, df: pd.DataFrame, sensor_columns: list[str]
    ) -> pd.DataFrame:
        """Apply sensor fusion techniques."""
        if not self.config["sensor_fusion"]["enable"]:
            return df

        correlation_threshold = self.config["sensor_fusion"]["correlation_threshold"]
        fusion_info = {}

        # Find highly correlated sensors
        if len(sensor_columns) > 1:
            sensor_data = df[sensor_columns]
            corr_matrix = sensor_data.corr().abs()

            # Find sensor pairs with high correlation
            high_corr_pairs = []
            for i, sensor1 in enumerate(sensor_columns):
                for sensor2 in sensor_columns[i + 1 :]:
                    if (
                        sensor1 in corr_matrix.columns
                        and sensor2 in corr_matrix.columns
                    ):
                        correlation = corr_matrix.loc[sensor1, sensor2]
                        if correlation > correlation_threshold:
                            high_corr_pairs.append((sensor1, sensor2, correlation))

            # Create fused features for highly correlated sensors
            for sensor1, sensor2, correlation in high_corr_pairs:
                # Weighted average based on data quality
                sensor1_quality = 1 - df[sensor1].isnull().sum() / len(df)
                sensor2_quality = 1 - df[sensor2].isnull().sum() / len(df)

                total_quality = sensor1_quality + sensor2_quality
                weight1 = sensor1_quality / total_quality
                weight2 = sensor2_quality / total_quality

                # Create fused sensor
                fused_name = f"{sensor1}_{sensor2}_fused"
                df[fused_name] = weight1 * df[sensor1] + weight2 * df[sensor2]

                fusion_info[fused_name] = {
                    "sensors": [sensor1, sensor2],
                    "correlation": correlation,
                    "weights": [weight1, weight2],
                }

                # Optionally remove redundant sensors
                if self.config["sensor_fusion"]["redundancy_removal"]:
                    # Keep the sensor with better quality
                    if sensor1_quality < sensor2_quality:
                        df = df.drop(columns=[sensor1])
                    else:
                        df = df.drop(columns=[sensor2])

        if fusion_info:
            self.preprocessing_steps.append(
                {"step": "sensor_fusion", "fused_sensors": fusion_info}
            )

        return df

    def _scale_features(
        self, df: pd.DataFrame, original_sensor_columns: list[str]
    ) -> pd.DataFrame:
        """Scale features with IoT-specific considerations."""
        method = self.config["scaling"]["method"]
        per_sensor = self.config["scaling"]["per_sensor"]

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if per_sensor and len(original_sensor_columns) > 1:
            # Scale sensor groups separately
            scaling_info = {}

            # Group features by sensor origin
            sensor_groups = {}
            for col in numeric_columns:
                # Find which sensor this feature belongs to
                parent_sensor = None
                for sensor in original_sensor_columns:
                    if col.startswith(sensor):
                        parent_sensor = sensor
                        break

                if parent_sensor:
                    if parent_sensor not in sensor_groups:
                        sensor_groups[parent_sensor] = []
                    sensor_groups[parent_sensor].append(col)
                else:
                    # Features not belonging to specific sensors
                    if "general" not in sensor_groups:
                        sensor_groups["general"] = []
                    sensor_groups["general"].append(col)

            # Scale each group
            for group_name, features in sensor_groups.items():
                if len(features) > 0:
                    if method == "standard":
                        scaler = StandardScaler()
                    elif method == "minmax":
                        scaler = MinMaxScaler(
                            feature_range=self.config["scaling"]["feature_range"]
                        )
                    elif method == "robust":
                        scaler = RobustScaler()

                    df[features] = scaler.fit_transform(df[features])
                    self.scalers[group_name] = scaler
                    scaling_info[group_name] = len(features)

            self.preprocessing_steps.append(
                {
                    "step": "feature_scaling_per_sensor",
                    "method": method,
                    "sensor_groups": scaling_info,
                }
            )

        else:
            # Scale all numeric features together
            if len(numeric_columns) > 0:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler(
                        feature_range=self.config["scaling"]["feature_range"]
                    )
                elif method == "robust":
                    scaler = RobustScaler()

                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                self.scalers["all"] = scaler

                self.preprocessing_steps.append(
                    {
                        "step": "feature_scaling_global",
                        "method": method,
                        "features_scaled": len(numeric_columns),
                    }
                )

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select most relevant features for anomaly detection."""
        original_features = len(df.columns)

        # Remove low variance features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit_transform(numeric_df)

            # Get selected column names
            selected_columns = numeric_df.columns[selector.get_support()]

            # Keep non-numeric columns and selected numeric columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            df = df[list(non_numeric_columns) + list(selected_columns)]

            self.feature_selectors["variance"] = selector

        # Remove highly correlated features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > 0.95)
            ]

            df = df.drop(columns=to_drop)

            if to_drop:
                self.preprocessing_steps.append(
                    {
                        "step": "correlation_feature_removal",
                        "threshold": 0.95,
                        "removed_features": len(to_drop),
                    }
                )

        self.preprocessing_steps.append(
            {
                "step": "feature_selection_summary",
                "original_features": original_features,
                "final_features": len(df.columns),
                "features_removed": original_features - len(df.columns),
            }
        )

        return df

    def _final_validation(
        self, df: pd.DataFrame, original_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Perform final validation of processed IoT data."""
        validation_results = {
            "shape_change": f"{original_shape} -> {df.shape}",
            "missing_values": df.isnull().sum().sum(),
            "infinite_values": np.isinf(df.select_dtypes(include=[np.number]))
            .sum()
            .sum(),
            "temporal_continuity": self._check_temporal_continuity(df),
            "data_types": dict(df.dtypes.astype(str)),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "processing_success": True,
        }

        # Check for any remaining issues
        issues = []
        if validation_results["missing_values"] > 0:
            issues.append("Missing values still present")
        if validation_results["infinite_values"] > 0:
            issues.append("Infinite values detected")
        if not validation_results["temporal_continuity"]:
            issues.append("Temporal continuity issues detected")

        validation_results["issues"] = issues
        validation_results["processing_success"] = len(issues) == 0

        return validation_results

    def _check_temporal_continuity(self, df: pd.DataFrame) -> bool:
        """Check if temporal data has reasonable continuity."""
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            # Check for reasonable time intervals
            time_diffs = df.index.to_series().diff().dropna()
            median_interval = time_diffs.median()

            # Allow some variation in sampling intervals
            acceptable_range = (median_interval * 0.5, median_interval * 3)
            irregular_intervals = (
                (time_diffs < acceptable_range[0]) | (time_diffs > acceptable_range[1])
            ).sum()

            # Consider continuity good if <10% irregular intervals
            return irregular_intervals / len(time_diffs) < 0.1

        return True  # Cannot assess if not temporal

    def _log_step(self, step_name: str):
        """Log preprocessing step."""
        if self.verbose:
            logger.info(f"Executing: {step_name}")

    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline configuration and fitted components."""
        pipeline_data = {
            "config": self.config,
            "preprocessing_steps": self.preprocessing_steps,
            "sensor_metadata": self.sensor_metadata,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load a saved preprocessing pipeline configuration."""
        with open(filepath) as f:
            pipeline_data = json.load(f)

        self.config = pipeline_data["config"]
        self.preprocessing_steps = pipeline_data.get("preprocessing_steps", [])
        self.sensor_metadata = pipeline_data.get("sensor_metadata", {})

        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Example usage of the IoT Sensor Preprocessor."""
    # Create sample IoT sensor data
    np.random.seed(42)
    n_samples = 5000

    # Generate synthetic sensor data with temporal patterns
    timestamps = pd.date_range("2023-01-01", periods=n_samples, freq="1min")

    # Create multiple sensors with different characteristics
    data = {
        "timestamp": timestamps,
        "temperature": 20
        + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 1440)
        + np.random.normal(0, 1, n_samples),  # Daily pattern
        "humidity": 60
        + 10 * np.cos(np.arange(n_samples) * 2 * np.pi / 1440)
        + np.random.normal(0, 2, n_samples),  # Daily pattern
        "pressure": 1013 + np.random.normal(0, 5, n_samples),  # Random walk
        "light_level": np.maximum(
            0,
            100 * (np.sin(np.arange(n_samples) * 2 * np.pi / 1440) + 1)
            + np.random.normal(0, 10, n_samples),
        ),  # Day/night
        "co2_level": 400
        + 100 * np.random.exponential(0.1, n_samples),  # Exponential distribution
    }

    df = pd.DataFrame(data)

    # Add some missing values and outliers
    df.loc[np.random.choice(df.index, 200), "temperature"] = np.nan
    df.loc[np.random.choice(df.index, 50), "pressure"] = (
        df["pressure"].iloc[np.random.choice(df.index, 50)] + 200
    )  # Outliers

    print("Original Data Shape:", df.shape)
    print("\nOriginal Data Info:")
    print(df.info())

    # Initialize preprocessor with custom config
    config = {
        "temporal": {
            "resample_frequency": "5min",  # Resample to 5-minute intervals
            "interpolation_method": "linear",
        },
        "missing_values": {"strategy": "interpolation", "max_gap_minutes": 15},
        "outliers": {"method": "iqr_per_sensor", "action": "interpolate"},
        "feature_engineering": {
            "rolling_statistics": True,
            "window_sizes": [3, 6, 12],  # 15min, 30min, 1hour windows
            "lag_features": True,
            "max_lags": 3,
        },
        "sensor_fusion": {"enable": True, "correlation_threshold": 0.7},
    }

    preprocessor = IoTSensorPreprocessor(config=config, verbose=True)

    # Apply preprocessing
    sensor_columns = ["temperature", "humidity", "pressure", "light_level", "co2_level"]
    processed_df, metadata = preprocessor.preprocess(df, "timestamp", sensor_columns)

    print(f"\nProcessed Data Shape: {processed_df.shape}")
    print("\nPreprocessing Steps Applied:")
    for i, step in enumerate(metadata["preprocessing_steps"], 1):
        print(f"{i}. {step['step']}")

    print("\nSensor Quality Assessment:")
    for sensor, stats in metadata["quality_results"]["sensor_analysis"].items():
        print(
            f"- {sensor}: Missing={stats['missing_ratio']:.3f}, Outliers={stats['outlier_ratio']:.3f}"
        )

    print("\nValidation Results:")
    print(f"- Processing Success: {metadata['final_validation']['processing_success']}")
    print(f"- Missing Values: {metadata['final_validation']['missing_values']}")
    print(f"- Memory Usage: {metadata['final_validation']['memory_usage_mb']:.2f} MB")

    # Save pipeline for reuse
    preprocessor.save_pipeline("iot_preprocessing_pipeline.json")

    print("\nIoT sensor preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
