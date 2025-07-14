#!/usr/bin/env python3
"""
Time Series Anomaly Detection Template

This template provides a comprehensive framework for detecting anomalies in time series data,
including seasonal decomposition, trend analysis, and context-aware anomaly detection.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Time series specific libraries
import os

# Pynomaly imports (adjust path as needed)
import sys

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


class TimeSeriesAnomalyDetector:
    """
    Comprehensive time series anomaly detection with seasonal handling and context awareness.

    Features:
    - Seasonal decomposition and trend analysis
    - Multiple anomaly types: point, contextual, collective
    - Sliding window analysis
    - Statistical and ML-based detection
    - Visualization and reporting
    """

    def __init__(
        self,
        contamination_rate: float = 0.05,
        seasonal_periods: int = None,
        window_size: int = 100,
        algorithm: str = "IsolationForest",
        random_state: int = 42,
    ):
        """
        Initialize the time series anomaly detector.

        Args:
            contamination_rate: Expected proportion of anomalies
            seasonal_periods: Number of periods in a season (auto-detect if None)
            window_size: Size of sliding window for analysis
            algorithm: Anomaly detection algorithm to use
            random_state: Random seed for reproducibility
        """
        self.contamination_rate = contamination_rate
        self.seasonal_periods = seasonal_periods
        self.window_size = window_size
        self.algorithm = algorithm
        self.random_state = random_state

        # Initialize components
        self.scaler = StandardScaler()
        self.detector = None
        self.decomposition = None
        self.results = {}

    def load_data(self, data_source, timestamp_col="timestamp", value_col="value"):
        """
        Load time series data from various sources.

        Args:
            data_source: File path, DataFrame, or data generator
            timestamp_col: Name of timestamp column
            value_col: Name of value column

        Returns:
            Processed DataFrame with datetime index
        """
        if isinstance(data_source, str):
            # Load from file
            if data_source.endswith(".csv"):
                df = pd.read_csv(data_source)
            elif data_source.endswith(".json"):
                df = pd.read_json(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise ValueError("Data source must be file path or DataFrame")

        # Convert timestamp column to datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

        # Ensure we have the value column
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in data")

        # Store original data
        self.original_data = df
        self.value_col = value_col
        self.timestamp_col = timestamp_col

        print(f"✅ Loaded time series data: {len(df)} points")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"📊 Value range: {df[value_col].min():.2f} to {df[value_col].max():.2f}")

        return df

    def analyze_time_series_properties(self, df=None):
        """
        Analyze basic time series properties and stationarity.

        Args:
            df: DataFrame to analyze (uses loaded data if None)

        Returns:
            Dictionary with analysis results
        """
        if df is None:
            df = self.original_data

        series = df[self.value_col]
        analysis = {}

        print("=== TIME SERIES ANALYSIS ===")

        # Basic statistics
        analysis["basic_stats"] = {
            "count": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing_values": series.isnull().sum(),
        }

        print("📊 Basic Statistics:")
        for key, value in analysis["basic_stats"].items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")

        # Frequency analysis
        time_diff = df.index.to_series().diff().dropna()
        most_common_freq = (
            time_diff.mode().iloc[0] if not time_diff.mode().empty else None
        )

        analysis["frequency"] = {
            "most_common_interval": most_common_freq,
            "irregular_intervals": (time_diff != most_common_freq).sum(),
            "missing_timestamps": (
                len(
                    pd.date_range(df.index.min(), df.index.max(), freq=most_common_freq)
                )
                - len(df)
                if most_common_freq
                else None
            ),
        }

        print("\n⏰ Frequency Analysis:")
        print(f"  • Most common interval: {most_common_freq}")
        print(
            f"  • Irregular intervals: {analysis['frequency']['irregular_intervals']}"
        )

        # Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(series.dropna())
            analysis["stationarity"] = {
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "is_stationary": adf_result[1] < 0.05,
                "critical_values": adf_result[4],
            }

            print("\n📈 Stationarity Test (ADF):")
            print(f"  • Test statistic: {adf_result[0]:.4f}")
            print(f"  • P-value: {adf_result[1]:.4f}")
            print(f"  • Is stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")

        except Exception as e:
            print(f"⚠️ Stationarity test failed: {str(e)}")
            analysis["stationarity"] = None

        # Auto-detect seasonality
        if self.seasonal_periods is None:
            try:
                # Try common seasonal periods
                seasonal_candidates = [7, 24, 30, 365]  # daily, hourly, monthly, yearly
                best_seasonal = self._detect_seasonality(series, seasonal_candidates)
                self.seasonal_periods = best_seasonal
                analysis["auto_seasonality"] = best_seasonal
                print(f"\n🔄 Auto-detected seasonal period: {best_seasonal}")
            except Exception as e:
                print(f"⚠️ Seasonality detection failed: {str(e)}")
                self.seasonal_periods = None

        return analysis

    def _detect_seasonality(self, series, candidates):
        """
        Detect the best seasonal period using autocorrelation.

        Args:
            series: Time series data
            candidates: List of potential seasonal periods

        Returns:
            Best seasonal period
        """
        best_period = None
        best_score = 0

        for period in candidates:
            if period < len(series) // 2:  # Need at least 2 full cycles
                try:
                    # Calculate autocorrelation at this lag
                    autocorr = series.autocorr(lag=period)
                    if not np.isnan(autocorr) and autocorr > best_score:
                        best_score = autocorr
                        best_period = period
                except:
                    continue

        return best_period

    def decompose_time_series(self, df=None, model="additive"):
        """
        Perform seasonal decomposition of the time series.

        Args:
            df: DataFrame to decompose (uses loaded data if None)
            model: 'additive' or 'multiplicative'

        Returns:
            Decomposition result object
        """
        if df is None:
            df = self.original_data

        if self.seasonal_periods is None or self.seasonal_periods < 2:
            print("⚠️ Skipping decomposition: no seasonal period detected")
            return None

        try:
            print(f"🔧 Performing {model} seasonal decomposition...")

            series = df[self.value_col]
            self.decomposition = seasonal_decompose(
                series,
                model=model,
                period=self.seasonal_periods,
                extrapolate_trend="freq",
            )

            print("✅ Decomposition complete")
            print(
                f"  • Trend component: {len(self.decomposition.trend.dropna())} points"
            )
            print(
                f"  • Seasonal component: {len(self.decomposition.seasonal.dropna())} points"
            )
            print(
                f"  • Residual component: {len(self.decomposition.resid.dropna())} points"
            )

            return self.decomposition

        except Exception as e:
            print(f"❌ Decomposition failed: {str(e)}")
            return None

    def engineer_features(self, df=None, include_decomposition=True):
        """
        Create comprehensive features for anomaly detection.

        Args:
            df: DataFrame to process (uses loaded data if None)
            include_decomposition: Whether to include decomposition features

        Returns:
            DataFrame with engineered features
        """
        if df is None:
            df = self.original_data

        print("🔧 Engineering time series features...")

        features_df = df.copy()
        series = df[self.value_col]

        # Basic statistical features
        features_df["value"] = series
        features_df["value_log"] = np.log1p(np.abs(series))
        features_df["value_diff"] = series.diff()
        features_df["value_pct_change"] = series.pct_change()

        # Rolling window features
        windows = [3, 7, 14, 30] if len(df) > 30 else [3, 5]

        for window in windows:
            if window < len(df):
                features_df[f"rolling_mean_{window}"] = series.rolling(window).mean()
                features_df[f"rolling_std_{window}"] = series.rolling(window).std()
                features_df[f"rolling_min_{window}"] = series.rolling(window).min()
                features_df[f"rolling_max_{window}"] = series.rolling(window).max()
                features_df[f"rolling_median_{window}"] = series.rolling(
                    window
                ).median()

                # Deviation from rolling statistics
                features_df[f"dev_from_mean_{window}"] = (
                    series - features_df[f"rolling_mean_{window}"]
                ).abs()
                features_df[f"zscore_{window}"] = (
                    series - features_df[f"rolling_mean_{window}"]
                ) / features_df[f"rolling_std_{window}"]

        # Lag features
        lags = [1, 2, 3, 7] if len(df) > 7 else [1, 2]
        for lag in lags:
            if lag < len(df):
                features_df[f"lag_{lag}"] = series.shift(lag)

        # Time-based features
        features_df["hour"] = df.index.hour
        features_df["day_of_week"] = df.index.dayofweek
        features_df["day_of_month"] = df.index.day
        features_df["month"] = df.index.month
        features_df["quarter"] = df.index.quarter
        features_df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Cyclical encoding for time features
        features_df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        features_df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        features_df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features_df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        features_df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        features_df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        # Decomposition features\n        if include_decomposition and self.decomposition is not None:\n            features_df['trend'] = self.decomposition.trend\n            features_df['seasonal'] = self.decomposition.seasonal\n            features_df['residual'] = self.decomposition.resid\n            \n            # Relative deviations\n            features_df['residual_abs'] = np.abs(features_df['residual'])\n            features_df['trend_change'] = features_df['trend'].diff()\n            features_df['seasonal_strength'] = np.abs(features_df['seasonal'])\n        \n        # Additional anomaly indicators\n        features_df['is_extreme'] = (np.abs(zscore(series.dropna())) > 3).astype(int)\n        features_df['local_outlier_factor'] = self._calculate_local_outliers(series)\n        \n        # Remove infinite and NaN values\n        features_df = features_df.replace([np.inf, -np.inf], np.nan)\n        \n        print(f\"✅ Feature engineering complete: {features_df.shape[1]} features created\")\n        \n        return features_df\n    \n    def _calculate_local_outliers(self, series, window_size=None):\n        \"\"\"Calculate local outlier scores using sliding window.\"\"\"\n        if window_size is None:\n            window_size = min(self.window_size, len(series) // 4)\n        \n        outlier_scores = np.zeros(len(series))\n        \n        for i in range(len(series)):\n            start_idx = max(0, i - window_size // 2)\n            end_idx = min(len(series), i + window_size // 2 + 1)\n            \n            window_data = series.iloc[start_idx:end_idx]\n            if len(window_data) > 3:\n                mean_val = window_data.mean()\n                std_val = window_data.std()\n                if std_val > 0:\n                    outlier_scores[i] = abs(series.iloc[i] - mean_val) / std_val\n        \n        return outlier_scores\n    \n    def detect_anomalies(self, features_df=None, method='ml'):\n        \"\"\"\n        Detect anomalies using specified method.\n        \n        Args:\n            features_df: DataFrame with features (auto-generated if None)\n            method: 'ml', 'statistical', or 'hybrid'\n            \n        Returns:\n            Dictionary with detection results\n        \"\"\"\n        if features_df is None:\n            features_df = self.engineer_features()\n        \n        print(f\"🔍 Detecting anomalies using {method} method...\")\n        \n        # Prepare feature matrix\n        feature_columns = [col for col in features_df.columns \n                          if col not in [self.value_col, 'is_extreme']]\n        X = features_df[feature_columns].fillna(method='ffill').fillna(method='bfill')\n        \n        if method == 'ml':\n            results = self._ml_anomaly_detection(X, features_df)\n        elif method == 'statistical':\n            results = self._statistical_anomaly_detection(features_df)\n        elif method == 'hybrid':\n            ml_results = self._ml_anomaly_detection(X, features_df)\n            stat_results = self._statistical_anomaly_detection(features_df)\n            results = self._combine_results(ml_results, stat_results)\n        else:\n            raise ValueError(f\"Unknown method: {method}\")\n        \n        # Store results\n        self.results = results\n        \n        print(f\"✅ Anomaly detection complete\")\n        print(f\"  • Total anomalies detected: {results['anomaly_count']}\")\n        print(f\"  • Anomaly rate: {results['anomaly_rate']:.1%}\")\n        \n        return results\n    \n    def _ml_anomaly_detection(self, X, features_df):\n        \"\"\"Machine learning based anomaly detection.\"\"\"\n        # Scale features\n        X_scaled = self.scaler.fit_transform(X)\n        \n        # Create Pynomaly dataset\n        dataset = Dataset(\n            name=\"time_series_features\",\n            data=X_scaled,\n            feature_names=list(X.columns)\n        )\n        \n        # Initialize detector\n        contamination_rate = ContaminationRate(self.contamination_rate)\n        detector = SklearnAdapter(self.algorithm, contamination_rate=contamination_rate)\n        \n        # Fit and detect\n        result = detector.fit_detect(dataset)\n        \n        # Extract results\n        scores = np.array([score.value for score in result.scores])\n        threshold = np.percentile(scores, (1 - self.contamination_rate) * 100)\n        anomaly_labels = (scores > threshold).astype(int)\n        \n        return {\n            'method': 'ml',\n            'algorithm': self.algorithm,\n            'anomaly_scores': scores,\n            'anomaly_labels': anomaly_labels,\n            'threshold': threshold,\n            'anomaly_count': anomaly_labels.sum(),\n            'anomaly_rate': anomaly_labels.mean(),\n            'feature_importance': None  # Could be added for tree-based methods\n        }\n    \n    def _statistical_anomaly_detection(self, features_df):\n        \"\"\"Statistical anomaly detection based on time series properties.\"\"\"\n        series = features_df[self.value_col]\n        \n        # Multiple statistical approaches\n        anomaly_indicators = []\n        \n        # 1. Z-score based detection\n        z_scores = np.abs(zscore(series.dropna()))\n        z_anomalies = (z_scores > 3).astype(int)\n        anomaly_indicators.append(z_anomalies)\n        \n        # 2. IQR based detection\n        Q1 = series.quantile(0.25)\n        Q3 = series.quantile(0.75)\n        IQR = Q3 - Q1\n        lower_bound = Q1 - 1.5 * IQR\n        upper_bound = Q3 + 1.5 * IQR\n        iqr_anomalies = ((series < lower_bound) | (series > upper_bound)).astype(int)\n        anomaly_indicators.append(iqr_anomalies)\n        \n        # 3. Residual based detection (if decomposition available)\n        if self.decomposition is not None:\n            residuals = self.decomposition.resid.dropna()\n            if len(residuals) > 0:\n                residual_threshold = residuals.std() * 2.5\n                residual_anomalies = (np.abs(residuals) > residual_threshold).astype(int)\n                # Align with original series\n                aligned_residual_anomalies = np.zeros(len(series))\n                aligned_residual_anomalies[residuals.index.get_indexer(series.index)] = residual_anomalies\n                anomaly_indicators.append(aligned_residual_anomalies)\n        \n        # 4. Rate of change detection\n        pct_change = series.pct_change().fillna(0)\n        change_threshold = pct_change.std() * 3\n        change_anomalies = (np.abs(pct_change) > change_threshold).astype(int)\n        anomaly_indicators.append(change_anomalies)\n        \n        # Combine indicators (majority vote)\n        if anomaly_indicators:\n            # Ensure all indicators have the same length\n            min_length = min(len(indicator) for indicator in anomaly_indicators)\n            aligned_indicators = [indicator[:min_length] for indicator in anomaly_indicators]\n            \n            combined_scores = np.mean(aligned_indicators, axis=0)\n            anomaly_labels = (combined_scores > 0.5).astype(int)\n        else:\n            combined_scores = np.zeros(len(series))\n            anomaly_labels = np.zeros(len(series), dtype=int)\n        \n        return {\n            'method': 'statistical',\n            'anomaly_scores': combined_scores,\n            'anomaly_labels': anomaly_labels,\n            'threshold': 0.5,\n            'anomaly_count': anomaly_labels.sum(),\n            'anomaly_rate': anomaly_labels.mean(),\n            'component_scores': {\n                'z_score': z_anomalies if len(anomaly_indicators) > 0 else None,\n                'iqr': iqr_anomalies if len(anomaly_indicators) > 1 else None,\n                'residual': aligned_residual_anomalies if 'aligned_residual_anomalies' in locals() else None,\n                'change_rate': change_anomalies if len(anomaly_indicators) > 3 else None\n            }\n        }\n    \n    def _combine_results(self, ml_results, stat_results):\n        \"\"\"Combine ML and statistical results.\"\"\"\n        # Weighted combination\n        ml_weight = 0.6\n        stat_weight = 0.4\n        \n        combined_scores = (ml_weight * ml_results['anomaly_scores'] + \n                          stat_weight * stat_results['anomaly_scores'])\n        \n        threshold = ml_weight * ml_results['threshold'] + stat_weight * stat_results['threshold']\n        anomaly_labels = (combined_scores > threshold).astype(int)\n        \n        return {\n            'method': 'hybrid',\n            'anomaly_scores': combined_scores,\n            'anomaly_labels': anomaly_labels,\n            'threshold': threshold,\n            'anomaly_count': anomaly_labels.sum(),\n            'anomaly_rate': anomaly_labels.mean(),\n            'ml_component': ml_results,\n            'statistical_component': stat_results\n        }\n    \n    def visualize_results(self, features_df=None, results=None, save_path=None):\n        \"\"\"Create comprehensive visualizations of the analysis.\"\"\"\n        if features_df is None:\n            features_df = self.engineer_features()\n        if results is None:\n            results = self.results\n        \n        if not results:\n            print(\"❌ No results to visualize. Run detect_anomalies() first.\")\n            return\n        \n        fig = plt.figure(figsize=(20, 16))\n        \n        # Main time series with anomalies\n        ax1 = plt.subplot(3, 2, 1)\n        series = features_df[self.value_col]\n        anomaly_mask = results['anomaly_labels'].astype(bool)\n        \n        # Plot normal points\n        normal_mask = ~anomaly_mask\n        ax1.plot(series.index[normal_mask], series.iloc[normal_mask], \n                'b-', alpha=0.7, label='Normal', linewidth=1)\n        \n        # Plot anomalies\n        if anomaly_mask.any():\n            ax1.scatter(series.index[anomaly_mask], series.iloc[anomaly_mask], \n                       c='red', s=50, alpha=0.8, label='Anomaly', zorder=5)\n        \n        ax1.set_title('Time Series with Detected Anomalies', fontsize=14, fontweight='bold')\n        ax1.set_ylabel('Value')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # Anomaly scores\n        ax2 = plt.subplot(3, 2, 2)\n        ax2.plot(series.index, results['anomaly_scores'], 'g-', alpha=0.7, linewidth=1)\n        ax2.axhline(y=results['threshold'], color='red', linestyle='--', \n                   label=f\"Threshold ({results['threshold']:.3f})\")\n        ax2.set_title('Anomaly Scores', fontsize=14, fontweight='bold')\n        ax2.set_ylabel('Score')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n        \n        # Seasonal decomposition (if available)\n        if self.decomposition is not None:\n            ax3 = plt.subplot(3, 2, 3)\n            ax3.plot(self.decomposition.trend.index, self.decomposition.trend, 'b-', alpha=0.8)\n            ax3.set_title('Trend Component', fontsize=14, fontweight='bold')\n            ax3.set_ylabel('Trend')\n            ax3.grid(True, alpha=0.3)\n            \n            ax4 = plt.subplot(3, 2, 4)\n            ax4.plot(self.decomposition.seasonal.index, self.decomposition.seasonal, 'orange', alpha=0.8)\n            ax4.set_title('Seasonal Component', fontsize=14, fontweight='bold')\n            ax4.set_ylabel('Seasonal')\n            ax4.grid(True, alpha=0.3)\n            \n            ax5 = plt.subplot(3, 2, 5)\n            ax5.plot(self.decomposition.resid.index, self.decomposition.resid, 'gray', alpha=0.8)\n            if anomaly_mask.any():\n                residual_anomalies = self.decomposition.resid.iloc[anomaly_mask[:len(self.decomposition.resid)]]\n                ax5.scatter(residual_anomalies.index, residual_anomalies, c='red', s=30, alpha=0.8)\n            ax5.set_title('Residual Component with Anomalies', fontsize=14, fontweight='bold')\n            ax5.set_ylabel('Residual')\n            ax5.grid(True, alpha=0.3)\n        else:\n            # Distribution analysis\n            ax3 = plt.subplot(3, 2, 3)\n            ax3.hist(series[normal_mask], bins=50, alpha=0.7, label='Normal', density=True)\n            if anomaly_mask.any():\n                ax3.hist(series[anomaly_mask], bins=20, alpha=0.7, label='Anomaly', density=True)\n            ax3.set_title('Value Distribution', fontsize=14, fontweight='bold')\n            ax3.set_xlabel('Value')\n            ax3.set_ylabel('Density')\n            ax3.legend()\n            ax3.grid(True, alpha=0.3)\n            \n            # Rolling statistics\n            ax4 = plt.subplot(3, 2, 4)\n            if 'rolling_mean_7' in features_df.columns:\n                ax4.plot(features_df.index, features_df['rolling_mean_7'], 'blue', alpha=0.8, label='7-day Mean')\n                if 'rolling_std_7' in features_df.columns:\n                    upper = features_df['rolling_mean_7'] + 2 * features_df['rolling_std_7']\n                    lower = features_df['rolling_mean_7'] - 2 * features_df['rolling_std_7']\n                    ax4.fill_between(features_df.index, lower, upper, alpha=0.2, color='blue')\n                if anomaly_mask.any():\n                    ax4.scatter(series.index[anomaly_mask], series.iloc[anomaly_mask], \n                               c='red', s=30, alpha=0.8)\n            ax4.set_title('Rolling Statistics with Anomalies', fontsize=14, fontweight='bold')\n            ax4.set_ylabel('Value')\n            ax4.legend()\n            ax4.grid(True, alpha=0.3)\n            \n            # Time-based analysis\n            ax5 = plt.subplot(3, 2, 5)\n            hourly_anomalies = pd.DataFrame({\n                'hour': features_df.index.hour,\n                'is_anomaly': anomaly_mask\n            }).groupby('hour')['is_anomaly'].mean()\n            \n            ax5.bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='coral')\n            ax5.set_title('Anomaly Rate by Hour', fontsize=14, fontweight='bold')\n            ax5.set_xlabel('Hour of Day')\n            ax5.set_ylabel('Anomaly Rate')\n            ax5.grid(True, alpha=0.3)\n        \n        # Method comparison (if hybrid)\n        ax6 = plt.subplot(3, 2, 6)\n        if results.get('method') == 'hybrid':\n            ax6.plot(series.index, results['ml_component']['anomaly_scores'], \n                    'blue', alpha=0.7, label='ML Scores')\n            ax6.plot(series.index, results['statistical_component']['anomaly_scores'], \n                    'green', alpha=0.7, label='Statistical Scores')\n            ax6.plot(series.index, results['anomaly_scores'], \n                    'red', alpha=0.8, label='Combined Scores')\n            ax6.set_title('Method Comparison (Hybrid)', fontsize=14, fontweight='bold')\n        else:\n            # Score distribution\n            ax6.hist(results['anomaly_scores'], bins=50, alpha=0.7, density=True)\n            ax6.axvline(x=results['threshold'], color='red', linestyle='--', \n                       label=f\"Threshold ({results['threshold']:.3f})\")\n            ax6.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')\n            ax6.set_xlabel('Score')\n            ax6.set_ylabel('Density')\n        \n        ax6.legend()\n        ax6.grid(True, alpha=0.3)\n        \n        plt.suptitle(f'Time Series Anomaly Detection Results - {results[\"method\"].title()} Method', \n                    fontsize=16, fontweight='bold')\n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            print(f\"📊 Visualization saved to: {save_path}\")\n        \n        plt.show()\n    \n    def generate_report(self, features_df=None, results=None, output_path=None):\n        \"\"\"Generate a comprehensive analysis report.\"\"\"\n        if features_df is None:\n            features_df = self.engineer_features()\n        if results is None:\n            results = self.results\n        \n        if not results:\n            print(\"❌ No results to report. Run detect_anomalies() first.\")\n            return\n        \n        report = []\n        report.append(\"# Time Series Anomaly Detection Report\\n\")\n        report.append(f\"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\")\n        report.append(f\"**Method**: {results['method'].title()}\\n\")\n        \n        # Data summary\n        report.append(\"## Data Summary\\n\")\n        report.append(f\"- **Total Points**: {len(features_df):,}\\n\")\n        report.append(f\"- **Date Range**: {features_df.index.min()} to {features_df.index.max()}\\n\")\n        report.append(f\"- **Value Range**: {features_df[self.value_col].min():.2f} to {features_df[self.value_col].max():.2f}\\n\")\n        \n        if self.seasonal_periods:\n            report.append(f\"- **Seasonal Period**: {self.seasonal_periods}\\n\")\n        \n        # Anomaly summary\n        report.append(\"## Anomaly Detection Results\\n\")\n        report.append(f\"- **Total Anomalies**: {results['anomaly_count']}\\n\")\n        report.append(f\"- **Anomaly Rate**: {results['anomaly_rate']:.1%}\\n\")\n        report.append(f\"- **Detection Threshold**: {results['threshold']:.4f}\\n\")\n        \n        # Method details\n        if results['method'] == 'ml':\n            report.append(f\"- **Algorithm**: {results['algorithm']}\\n\")\n        elif results['method'] == 'statistical':\n            report.append(f\"- **Components**: Z-score, IQR, Residual, Rate-of-change\\n\")\n        \n        # Time-based analysis\n        anomaly_mask = results['anomaly_labels'].astype(bool)\n        if anomaly_mask.any():\n            anomaly_times = features_df.index[anomaly_mask]\n            \n            report.append(\"## Temporal Analysis\\n\")\n            \n            # Hour distribution\n            hourly_dist = pd.Series(anomaly_times.hour).value_counts().sort_index()\n            report.append(f\"**Anomalies by Hour**:\\n\")\n            for hour, count in hourly_dist.head(5).items():\n                report.append(f\"- Hour {hour}: {count} anomalies\\n\")\n            \n            # Day of week distribution\n            dow_dist = pd.Series(anomaly_times.dayofweek).value_counts().sort_index()\n            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n            report.append(f\"\\n**Anomalies by Day of Week**:\\n\")\n            for dow, count in dow_dist.items():\n                report.append(f\"- {days[dow]}: {count} anomalies\\n\")\n        \n        # Top anomalies\n        if anomaly_mask.any():\n            report.append(\"## Top 10 Anomalies\\n\")\n            anomaly_scores_series = pd.Series(results['anomaly_scores'], index=features_df.index)\n            top_anomalies = anomaly_scores_series[anomaly_mask].nlargest(10)\n            \n            report.append(\"| Timestamp | Value | Score |\\n\")\n            report.append(\"|-----------|-------|-------|\\n\")\n            \n            for timestamp, score in top_anomalies.items():\n                value = features_df.loc[timestamp, self.value_col]\n                report.append(f\"| {timestamp} | {value:.2f} | {score:.4f} |\\n\")\n        \n        # Recommendations\n        report.append(\"## Recommendations\\n\")\n        \n        if results['anomaly_rate'] > 0.1:\n            report.append(\"- ⚠️ High anomaly rate detected (>10%). Consider adjusting threshold or investigating data quality.\\n\")\n        elif results['anomaly_rate'] < 0.01:\n            report.append(\"- 🔍 Low anomaly rate detected (<1%). Consider lowering threshold for more sensitive detection.\\n\")\n        else:\n            report.append(\"- ✅ Anomaly rate within expected range.\\n\")\n        \n        if self.decomposition is None:\n            report.append(\"- 📊 Consider enabling seasonal decomposition for better trend and seasonal analysis.\\n\")\n        \n        if results['method'] != 'hybrid':\n            report.append(\"- 🔄 Consider using hybrid method for more robust detection.\\n\")\n        \n        report_text = ''.join(report)\n        \n        if output_path:\n            with open(output_path, 'w') as f:\n                f.write(report_text)\n            print(f\"📄 Report saved to: {output_path}\")\n        \n        return report_text\n\n\n# Example usage and testing\nif __name__ == \"__main__\":\n    # Create sample time series data with anomalies\n    def generate_sample_data(n_points=1000, anomaly_rate=0.05):\n        \"\"\"Generate sample time series data with embedded anomalies.\"\"\"\n        np.random.seed(42)\n        \n        # Create time index\n        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')\n        \n        # Generate base signal with trend and seasonality\n        t = np.arange(n_points)\n        trend = 0.01 * t\n        daily_seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern\n        weekly_seasonal = 5 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern\n        noise = np.random.normal(0, 2, n_points)\n        \n        base_signal = 50 + trend + daily_seasonal + weekly_seasonal + noise\n        \n        # Add anomalies\n        n_anomalies = int(n_points * anomaly_rate)\n        anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)\n        \n        # Different types of anomalies\n        for idx in anomaly_indices:\n            anomaly_type = np.random.choice(['spike', 'dip', 'shift'])\n            if anomaly_type == 'spike':\n                base_signal[idx] += np.random.uniform(20, 40)\n            elif anomaly_type == 'dip':\n                base_signal[idx] -= np.random.uniform(20, 40)\n            else:  # shift\n                # Create a temporary shift\n                shift_length = min(5, n_points - idx)\n                shift_value = np.random.uniform(-15, 15)\n                base_signal[idx:idx+shift_length] += shift_value\n        \n        # Create DataFrame\n        df = pd.DataFrame({\n            'timestamp': dates,\n            'value': base_signal,\n            'is_anomaly': 0\n        })\n        df.loc[anomaly_indices, 'is_anomaly'] = 1\n        \n        return df\n    \n    print(\"🧪 Running Time Series Anomaly Detection Example\")\n    print(\"=\" * 55)\n    \n    # Generate sample data\n    print(\"\\n📊 Generating sample time series data...\")\n    sample_data = generate_sample_data(n_points=2000, anomaly_rate=0.03)\n    print(f\"Generated {len(sample_data)} data points with {sample_data['is_anomaly'].sum()} embedded anomalies\")\n    \n    # Initialize detector\n    detector = TimeSeriesAnomalyDetector(\n        contamination_rate=0.05,\n        algorithm=\"IsolationForest\",\n        window_size=50,\n        random_state=42\n    )\n    \n    # Load and analyze data\n    print(\"\\n🔄 Loading and analyzing data...\")\n    df = detector.load_data(sample_data, timestamp_col='timestamp', value_col='value')\n    \n    # Analyze properties\n    analysis = detector.analyze_time_series_properties()\n    \n    # Perform decomposition\n    print(\"\\n🔧 Performing seasonal decomposition...\")\n    decomposition = detector.decompose_time_series()\n    \n    # Engineer features\n    print(\"\\n⚙️ Engineering features...\")\n    features = detector.engineer_features()\n    print(f\"Feature matrix shape: {features.shape}\")\n    \n    # Detect anomalies using different methods\n    print(\"\\n🔍 Detecting anomalies...\")\n    \n    # ML method\n    ml_results = detector.detect_anomalies(method='ml')\n    \n    # Statistical method\n    stat_results = detector.detect_anomalies(method='statistical')\n    \n    # Hybrid method\n    hybrid_results = detector.detect_anomalies(method='hybrid')\n    \n    # Evaluate if ground truth is available\n    if 'is_anomaly' in sample_data.columns:\n        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score\n        \n        ground_truth = sample_data['is_anomaly'].values\n        \n        print(\"\\n📊 Evaluation Results:\")\n        \n        for method_name, results in [('ML', ml_results), ('Statistical', stat_results), ('Hybrid', hybrid_results)]:\n            predictions = results['anomaly_labels']\n            scores = results['anomaly_scores']\n            \n            # Align lengths if needed\n            min_length = min(len(ground_truth), len(predictions))\n            gt_aligned = ground_truth[:min_length]\n            pred_aligned = predictions[:min_length]\n            scores_aligned = scores[:min_length]\n            \n            precision = precision_score(gt_aligned, pred_aligned)\n            recall = recall_score(gt_aligned, pred_aligned)\n            f1 = f1_score(gt_aligned, pred_aligned)\n            auc = roc_auc_score(gt_aligned, scores_aligned)\n            \n            print(f\"\\n{method_name} Method:\")\n            print(f\"  • Precision: {precision:.4f}\")\n            print(f\"  • Recall: {recall:.4f}\")\n            print(f\"  • F1-Score: {f1:.4f}\")\n            print(f\"  • AUC-ROC: {auc:.4f}\")\n    \n    # Generate visualizations\n    print(\"\\n📊 Generating visualizations...\")\n    detector.visualize_results(features, hybrid_results)\n    \n    # Generate report\n    print(\"\\n📄 Generating report...\")\n    report = detector.generate_report(features, hybrid_results)\n    print(\"\\nReport Preview:\")\n    print(report[:500] + \"...\")\n    \n    print(\"\\n✅ Time series anomaly detection example completed!\")"
