#!/usr/bin/env python3
"""
Time Series Anomaly Detection Example
====================================

This example demonstrates anomaly detection in time series data,
including seasonal patterns, trend analysis, and temporal features.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pynomaly.infrastructure.config import create_container


def generate_time_series_data(n_points: int = 1000, anomaly_rate: float = 0.05):
    """Generate synthetic time series data with seasonal patterns and anomalies."""

    # Create time index
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_points)]

    # Generate base signal with trend and seasonality
    t = np.arange(n_points)

    # Components of the time series
    trend = 0.001 * t  # Slight upward trend
    daily_seasonal = 10 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    weekly_seasonal = 5 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle
    noise = np.random.normal(0, 2, n_points)

    # Base signal
    base_signal = 50 + trend + daily_seasonal + weekly_seasonal + noise

    # Add anomalies
    anomaly_indices = np.random.choice(
        n_points, int(n_points * anomaly_rate), replace=False
    )
    anomalous_signal = base_signal.copy()

    for idx in anomaly_indices:
        # Different types of anomalies
        anomaly_type = np.random.choice(["spike", "drop", "shift"])

        if anomaly_type == "spike":
            anomalous_signal[idx] += np.random.uniform(20, 40)
        elif anomaly_type == "drop":
            anomalous_signal[idx] -= np.random.uniform(20, 40)
        else:  # shift - affects multiple consecutive points
            shift_length = min(10, n_points - idx)
            anomalous_signal[idx : idx + shift_length] += np.random.uniform(15, 25)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "value": anomalous_signal,
            "is_anomaly": [i in anomaly_indices for i in range(n_points)],
        }
    )

    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from time series data."""

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month

    # Cyclical encoding for temporal features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f"rolling_mean_{window}"] = df["value"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["value"].rolling(window=window).std()
        df[f"rolling_min_{window}"] = df["value"].rolling(window=window).min()
        df[f"rolling_max_{window}"] = df["value"].rolling(window=window).max()

    # Rate of change
    df["rate_of_change"] = df["value"].diff()
    df["acceleration"] = df["rate_of_change"].diff()

    # Distance from moving average
    df["dist_from_ma_24"] = df["value"] - df["rolling_mean_24"]
    df["dist_from_ma_168"] = (
        df["value"] - df["value"].rolling(window=168).mean()
    )  # Weekly MA

    # Remove rows with NaN values (due to lag and rolling features)
    df = df.dropna().reset_index(drop=True)

    return df


async def basic_time_series_detection():
    """Basic time series anomaly detection example."""
    print("📈 Basic Time Series Anomaly Detection")
    print("=" * 50)

    # Generate time series data
    print("📊 Generating synthetic time series data...")
    raw_df = generate_time_series_data(n_points=1000, anomaly_rate=0.05)

    # Extract temporal features
    print("🔧 Extracting temporal features...")
    df = extract_temporal_features(raw_df)

    print(f"   - Original data points: {len(raw_df)}")
    print(f"   - After feature extraction: {len(df)}")
    print(
        f"   - Features created: {len(df.columns) - 3}"
    )  # Exclude timestamp, value, is_anomaly

    # Prepare data for detection
    feature_columns = [
        col for col in df.columns if col not in ["timestamp", "is_anomaly"]
    ]

    # Split into train/test
    split_point = int(len(df) * 0.7)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    # Create container and services
    container = create_container()
    dataset_service = container.dataset_service()
    detection_service = container.detection_service()

    # Create training dataset (only normal data)
    train_normal = train_df[~train_df["is_anomaly"]]
    train_data = train_normal[feature_columns].to_dict("records")

    print(f"🎯 Creating training dataset with {len(train_data)} normal samples...")
    dataset = await dataset_service.create_from_data(
        data=train_data,
        name="Time Series Training Data",
        description="Normal time series data for training",
    )

    # Create detector
    print("🔧 Creating IsolationForest detector for time series...")
    detector = await detection_service.create_detector(
        name="Time Series Anomaly Detector",
        algorithm="IsolationForest",
        parameters={"contamination": 0.1, "n_estimators": 100, "random_state": 42},
    )

    # Train detector
    print("🎯 Training detector...")
    await detection_service.train_detector(detector.id, dataset.id)

    # Test on holdout data
    print(f"🔍 Testing on {len(test_df)} test samples...")
    test_data = test_df[feature_columns].to_dict("records")

    predictions = []
    scores = []

    for i, data_point in enumerate(test_data):
        try:
            result = await detection_service.detect_single(detector.id, data_point)
            predictions.append(result.is_anomaly)
            scores.append(result.anomaly_score)
        except Exception as e:
            print(f"   Error detecting point {i}: {e}")
            predictions.append(False)
            scores.append(0.0)

    # Evaluate results
    test_df_subset = test_df.iloc[: len(predictions)].copy()
    test_df_subset["predicted"] = predictions
    test_df_subset["anomaly_score"] = scores

    # Calculate metrics
    true_positives = sum((test_df_subset["is_anomaly"]) & (test_df_subset["predicted"]))
    false_positives = sum(
        (~test_df_subset["is_anomaly"]) & (test_df_subset["predicted"])
    )
    true_negatives = sum(
        (~test_df_subset["is_anomaly"]) & (~test_df_subset["predicted"])
    )
    false_negatives = sum(
        (test_df_subset["is_anomaly"]) & (~test_df_subset["predicted"])
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (true_positives + true_negatives) / len(test_df_subset)

    print("\n📊 Detection Results:")
    print(f"   Accuracy:  {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1_score:.3f}")

    # Show some detected anomalies
    detected_anomalies = test_df_subset[test_df_subset["predicted"]].head(10)
    if len(detected_anomalies) > 0:
        print("\n🚨 Sample Detected Anomalies:")
        for _, row in detected_anomalies.iterrows():
            timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            value = row["value"]
            score = row["anomaly_score"]
            actual = "✓" if row["is_anomaly"] else "✗"
            print(
                f"   [{timestamp}] Value: {value:6.1f} | Score: {score:.3f} | Actual: {actual}"
            )


async def multi_algorithm_time_series():
    """Compare multiple algorithms on time series data."""
    print("\n🎭 Multi-Algorithm Time Series Comparison")
    print("=" * 50)

    # Generate data
    raw_df = generate_time_series_data(n_points=500, anomaly_rate=0.08)
    df = extract_temporal_features(raw_df)

    feature_columns = [
        col for col in df.columns if col not in ["timestamp", "is_anomaly"]
    ]

    # Split data
    split_point = int(len(df) * 0.7)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    # Training data (normal only)
    train_normal = train_df[~train_df["is_anomaly"]]
    train_data = train_normal[feature_columns].to_dict("records")

    container = create_container()
    dataset_service = container.dataset_service()
    detection_service = container.detection_service()

    # Create dataset
    dataset = await dataset_service.create_from_data(
        data=train_data, name="Multi-Algorithm Time Series Data"
    )

    # Test multiple algorithms
    algorithms = [
        ("IsolationForest", {"contamination": 0.1, "n_estimators": 50}),
        ("LOF", {"contamination": 0.1, "n_neighbors": 20}),
        ("OCSVM", {"contamination": 0.1, "kernel": "rbf"}),
        ("COPOD", {"contamination": 0.1}),
    ]

    results = {}

    for algo_name, params in algorithms:
        print(f"🔧 Testing {algo_name}...")

        try:
            # Create and train detector
            detector = await detection_service.create_detector(
                name=f"TS {algo_name}", algorithm=algo_name, parameters=params
            )
            await detection_service.train_detector(detector.id, dataset.id)

            # Test
            test_data = test_df[feature_columns].to_dict("records")
            predictions = []

            for data_point in test_data:
                try:
                    result = await detection_service.detect_single(
                        detector.id, data_point
                    )
                    predictions.append(result.is_anomaly)
                except:
                    predictions.append(False)

            # Calculate F1 score
            test_subset = test_df.iloc[: len(predictions)]
            tp = sum((test_subset["is_anomaly"]) & predictions)
            fp = sum((~test_subset["is_anomaly"]) & predictions)
            fn = sum((test_subset["is_anomaly"]) & (~np.array(predictions)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results[algo_name] = {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "detected_anomalies": sum(predictions),
            }

        except Exception as e:
            print(f"   ❌ Error with {algo_name}: {e}")
            results[algo_name] = {
                "f1_score": 0,
                "precision": 0,
                "recall": 0,
                "detected_anomalies": 0,
            }

    # Display comparison
    print("\n📊 Algorithm Comparison Results:")
    print("Algorithm        | F1 Score | Precision | Recall | Detected")
    print("-" * 60)

    for algo, metrics in results.items():
        f1 = metrics["f1_score"]
        prec = metrics["precision"]
        rec = metrics["recall"]
        det = metrics["detected_anomalies"]
        print(f"{algo:<15} | {f1:8.3f} | {prec:9.3f} | {rec:6.3f} | {det:8d}")


async def seasonal_decomposition_example():
    """Advanced example with seasonal decomposition."""
    print("\n🌊 Seasonal Decomposition for Anomaly Detection")
    print("=" * 50)

    # Generate more complex seasonal data
    n_points = 2000  # About 83 days of hourly data
    raw_df = generate_time_series_data(n_points=n_points, anomaly_rate=0.03)

    print(f"📊 Generated {n_points} data points with seasonal patterns")

    # Advanced feature extraction with seasonal decomposition
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Manual seasonal decomposition (simplified)
    # Daily seasonality
    df["hour"] = df["timestamp"].dt.hour
    hourly_means = df.groupby("hour")["value"].mean()
    df["daily_seasonal"] = df["hour"].map(hourly_means)

    # Weekly seasonality
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    daily_means = df.groupby("day_of_week")["value"].mean()
    df["weekly_seasonal"] = df["day_of_week"].map(daily_means)

    # Trend (simple moving average)
    df["trend"] = df["value"].rolling(window=168, center=True).mean()  # Weekly trend

    # Residual (what's left after removing trend and seasonality)
    df["residual"] = df["value"] - df["daily_seasonal"] - df["weekly_seasonal"]
    df["residual"] = df["residual"].fillna(df["residual"].mean())

    # Features for detection
    df["value_normalized"] = (df["value"] - df["value"].mean()) / df["value"].std()
    df["residual_normalized"] = (df["residual"] - df["residual"].mean()) / df[
        "residual"
    ].std()

    # Additional features
    df["value_lag_1"] = df["value"].shift(1)
    df["value_lag_24"] = df["value"].shift(24)  # Same hour yesterday
    df["residual_lag_1"] = df["residual"].shift(1)

    # Rolling statistics of residuals
    df["residual_rolling_std"] = df["residual"].rolling(window=24).std()
    df["residual_rolling_mean"] = df["residual"].rolling(window=24).mean()

    # Remove NaN values
    df = df.dropna().reset_index(drop=True)

    print(f"   After preprocessing: {len(df)} data points")

    # Feature selection for detection
    feature_columns = [
        "value_normalized",
        "residual_normalized",
        "value_lag_1",
        "value_lag_24",
        "residual_lag_1",
        "residual_rolling_std",
        "residual_rolling_mean",
    ]

    # Split and create dataset
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    # Training data (normal only)
    train_normal = train_df[~train_df["is_anomaly"]]
    train_data = train_normal[feature_columns].to_dict("records")

    container = create_container()
    dataset_service = container.dataset_service()
    detection_service = container.detection_service()

    print(f"🎯 Creating dataset with {len(train_data)} normal training samples...")
    dataset = await dataset_service.create_from_data(
        data=train_data, name="Seasonal Time Series Data"
    )

    # Create detector optimized for seasonal data
    print("🔧 Creating detector optimized for seasonal patterns...")
    detector = await detection_service.create_detector(
        name="Seasonal Anomaly Detector",
        algorithm="IsolationForest",
        parameters={
            "contamination": 0.05,
            "n_estimators": 200,
            "max_features": 0.8,
            "random_state": 42,
        },
    )

    await detection_service.train_detector(detector.id, dataset.id)

    # Test detection
    print(f"🔍 Testing on {len(test_df)} samples...")
    test_data = test_df[feature_columns].to_dict("records")

    predictions = []
    scores = []

    for data_point in test_data:
        try:
            result = await detection_service.detect_single(detector.id, data_point)
            predictions.append(result.is_anomaly)
            scores.append(result.anomaly_score)
        except:
            predictions.append(False)
            scores.append(0.0)

    # Results
    test_subset = test_df.iloc[: len(predictions)].copy()
    test_subset["predicted"] = predictions
    test_subset["anomaly_score"] = scores

    # Metrics
    tp = sum((test_subset["is_anomaly"]) & (test_subset["predicted"]))
    fp = sum((~test_subset["is_anomaly"]) & (test_subset["predicted"]))
    fn = sum((test_subset["is_anomaly"]) & (~test_subset["predicted"]))
    tn = sum((~test_subset["is_anomaly"]) & (~test_subset["predicted"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / len(test_subset)

    print("\n📊 Seasonal Detection Results:")
    print(f"   Accuracy:  {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   True Anomalies: {sum(test_subset['is_anomaly'])}")
    print(f"   Detected Anomalies: {sum(predictions)}")


if __name__ == "__main__":
    print("📈 Pynomaly Time Series Anomaly Detection Examples")
    print("=" * 60)

    # Run examples
    asyncio.run(basic_time_series_detection())
    asyncio.run(multi_algorithm_time_series())
    asyncio.run(seasonal_decomposition_example())

    print("\n✅ Time series examples completed!")
    print("\nKey takeaways:")
    print("- Temporal features are crucial for time series anomaly detection")
    print("- Seasonal decomposition helps isolate true anomalies")
    print("- Different algorithms perform better on different patterns")
    print("- Feature engineering significantly impacts performance")
