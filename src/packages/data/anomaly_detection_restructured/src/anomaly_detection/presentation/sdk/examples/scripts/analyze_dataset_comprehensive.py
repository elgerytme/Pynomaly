#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script for Pynomaly

Provides analysis examples for each type of dataset, demonstrating
different approaches, algorithms, and handling strategies.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for consistent plots
plt.style.use("default")
sns.set_palette("husl")


def load_dataset_with_metadata(dataset_name, base_dir="examples/sample_datasets"):
    """Load dataset and its metadata"""
    base_path = Path(base_dir)

    # Try synthetic first, then real_world
    for subdir in ["synthetic", "real_world"]:
        data_file = base_path / subdir / f"{dataset_name}.csv"
        metadata_file = base_path / subdir / f"{dataset_name}_metadata.json"

        if data_file.exists():
            df = pd.read_csv(data_file)

            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

            return df, metadata

    raise FileNotFoundError(f"Dataset {dataset_name} not found")


def analyze_financial_fraud():
    """Analysis approach for financial fraud detection"""
    print("=" * 80)
    print("FINANCIAL FRAUD DATASET ANALYSIS")
    print("=" * 80)

    # Load data
    df, metadata = load_dataset_with_metadata("financial_fraud")

    print(f"Dataset: {metadata.get('description', 'Financial fraud detection')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    # Basic statistics
    print("\n1. DATA CHARACTERISTICS")
    print("-" * 40)

    # Amount distribution analysis
    normal_amounts = df[df["is_anomaly"] == 0]["transaction_amount"]
    fraud_amounts = df[df["is_anomaly"] == 1]["transaction_amount"]

    print("Normal transaction amounts:")
    print(f"  Mean: ${normal_amounts.mean():.2f}")
    print(f"  Median: ${normal_amounts.median():.2f}")
    print(f"  95th percentile: ${normal_amounts.quantile(0.95):.2f}")

    print("Fraudulent transaction amounts:")
    print(f"  Mean: ${fraud_amounts.mean():.2f}")
    print(f"  Median: ${fraud_amounts.median():.2f}")
    print(f"  95th percentile: ${fraud_amounts.quantile(0.95):.2f}")

    # Feature importance insights
    print("\n2. KEY FRAUD INDICATORS")
    print("-" * 40)

    # Amount-based indicators
    large_amounts = df["transaction_amount"] > df["transaction_amount"].quantile(0.95)
    micro_amounts = df["transaction_amount"] < 1

    print(
        f"Large amounts (>95th percentile): {(large_amounts & (df['is_anomaly'] == 1)).sum()} fraud / {large_amounts.sum()} total"
    )
    print(
        f"Micro amounts (<$1): {(micro_amounts & (df['is_anomaly'] == 1)).sum()} fraud / {micro_amounts.sum()} total"
    )

    # Time-based indicators
    unusual_hours = df["hour_of_day"] < 6
    print(
        f"Unusual hours (midnight-6am): {(unusual_hours & (df['is_anomaly'] == 1)).sum()} fraud / {unusual_hours.sum()} total"
    )

    # Velocity indicators
    high_velocity = df["velocity_score"] > df["velocity_score"].quantile(0.95)
    print(
        f"High velocity (>95th percentile): {(high_velocity & (df['is_anomaly'] == 1)).sum()} fraud / {high_velocity.sum()} total"
    )

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Feature Engineering:")
    print("     - Use log-transformed amounts to handle skewness")
    print("     - Engineer time-based features (hour, day, weekend)")
    print("     - Create velocity and frequency-based features")
    print("     - Use categorical encoding for merchant types")

    print("  2. Algorithm Selection:")
    print("     - IsolationForest: Excellent for mixed numerical/categorical")
    print("     - LocalOutlierFactor: Good for density-based detection")
    print("     - OneClassSVM: Effective for non-linear boundaries")

    print("  3. Evaluation Strategy:")
    print("     - Focus on precision (minimize false positives)")
    print("     - Use cost-sensitive evaluation (fraud costs more)")
    print("     - Monitor for concept drift over time")

    return df


def analyze_network_intrusion():
    """Analysis approach for network intrusion detection"""
    print("=" * 80)
    print("NETWORK INTRUSION DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("network_intrusion")

    print(f"Dataset: {metadata.get('description', 'Network intrusion detection')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. TRAFFIC PATTERN ANALYSIS")
    print("-" * 40)

    # Protocol distribution
    protocol_dist = df.groupby(["protocol", "is_anomaly"]).size().unstack(fill_value=0)
    print("Protocol distribution:")
    print(protocol_dist)

    # Port analysis
    normal_ports = df[df["is_anomaly"] == 0]["destination_port"].value_counts().head(5)
    attack_ports = df[df["is_anomaly"] == 1]["destination_port"].value_counts().head(5)

    print(f"\nTop normal traffic ports: {list(normal_ports.index)}")
    print(f"Top attack traffic ports: {list(attack_ports.index)}")

    # Traffic volume analysis
    print("\n2. TRAFFIC VOLUME INDICATORS")
    print("-" * 40)

    normal_traffic = df[df["is_anomaly"] == 0]
    attack_traffic = df[df["is_anomaly"] == 1]

    print(
        f"Normal traffic packets/second: {normal_traffic['packets_per_second'].mean():.2f} ± {normal_traffic['packets_per_second'].std():.2f}"
    )
    print(
        f"Attack traffic packets/second: {attack_traffic['packets_per_second'].mean():.2f} ± {attack_traffic['packets_per_second'].std():.2f}"
    )

    print(
        f"Normal traffic bytes/packet: {normal_traffic['bytes_per_packet'].mean():.2f} ± {normal_traffic['bytes_per_packet'].std():.2f}"
    )
    print(
        f"Attack traffic bytes/packet: {attack_traffic['bytes_per_packet'].mean():.2f} ± {attack_traffic['bytes_per_packet'].std():.2f}"
    )

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Data Preprocessing:")
    print("     - Log-transform highly skewed features (packet counts, bytes)")
    print("     - One-hot encode categorical features (protocol)")
    print("     - Normalize traffic volume features")
    print("     - Handle port numbers (standard vs. high ports)")

    print("  2. Feature Engineering:")
    print("     - Traffic velocity features (packets/second, bytes/second)")
    print("     - Connection duration patterns")
    print("     - Port classification (well-known, registered, dynamic)")
    print("     - Protocol-specific features")

    print("  3. Detection Strategy:")
    print("     - Real-time streaming detection")
    print("     - Multi-level detection (connection, session, host)")
    print("     - Ensemble methods for different attack types")

    return df


def analyze_iot_sensors():
    """Analysis approach for IoT sensor anomaly detection"""
    print("=" * 80)
    print("IOT SENSOR DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("iot_sensors")

    print(f"Dataset: {metadata.get('description', 'IoT sensor monitoring')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. SENSOR READING ANALYSIS")
    print("-" * 40)

    sensor_cols = [
        "temperature_celsius",
        "humidity_percent",
        "pressure_hpa",
        "vibration_level",
    ]

    for col in sensor_cols:
        normal_values = df[df["is_anomaly"] == 0][col]
        anomaly_values = df[df["is_anomaly"] == 1][col]

        print(f"{col}:")
        print(
            f"  Normal: {normal_values.mean():.2f} ± {normal_values.std():.2f} (range: {normal_values.min():.2f} - {normal_values.max():.2f})"
        )
        print(
            f"  Anomaly: {anomaly_values.mean():.2f} ± {anomaly_values.std():.2f} (range: {anomaly_values.min():.2f} - {anomaly_values.max():.2f})"
        )

    print("\n2. TEMPORAL PATTERN ANALYSIS")
    print("-" * 40)

    # Daily patterns
    hourly_anomalies = df.groupby("hour_of_day")["is_anomaly"].agg(
        ["count", "sum", "mean"]
    )
    peak_anomaly_hours = hourly_anomalies.nlargest(3, "mean").index
    print(f"Peak anomaly hours: {list(peak_anomaly_hours)}")

    # Correlation analysis
    correlation_features = [
        "temp_humidity_ratio",
        "pressure_deviation",
        "vibration_log",
    ]
    print("\nFeature correlations with anomalies:")
    for feature in correlation_features:
        corr = df[feature].corr(df["is_anomaly"])
        print(f"  {feature}: {corr:.3f}")

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Time Series Considerations:")
    print("     - Account for seasonal patterns (daily, weekly)")
    print("     - Use sliding window features")
    print("     - Consider temporal dependencies")
    print("     - Handle drift in sensor calibration")

    print("  2. Sensor-Specific Features:")
    print("     - Cross-sensor correlation features")
    print("     - Rate of change features")
    print("     - Statistical features over time windows")
    print("     - Environmental correlation features")

    print("  3. Anomaly Types:")
    print("     - Point anomalies: Single bad readings")
    print("     - Contextual anomalies: Values unusual for time/conditions")
    print("     - Collective anomalies: Patterns over time")
    print("     - Sensor failures: Stuck values, impossible readings")

    return df


def analyze_manufacturing_quality():
    """Analysis approach for manufacturing quality control"""
    print("=" * 80)
    print("MANUFACTURING QUALITY DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("manufacturing_quality")

    print(f"Dataset: {metadata.get('description', 'Manufacturing quality control')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. SPECIFICATION ANALYSIS")
    print("-" * 40)

    # Dimension specifications
    spec_limits = {
        "dimension_1_mm": (98, 102),  # ±2mm tolerance
        "dimension_2_mm": (49, 51),  # ±1mm tolerance
        "weight_grams": (480, 520),  # ±20g tolerance
        "hardness_hrc": (55, 65),  # ±5 HRC tolerance
    }

    for feature, (lower, upper) in spec_limits.items():
        if feature in df.columns:
            out_of_spec = (df[feature] < lower) | (df[feature] > upper)
            defect_rate = out_of_spec.mean()
            actual_defects = (out_of_spec & (df["is_anomaly"] == 1)).sum()

            print(f"{feature}:")
            print(f"  Specification: {lower} - {upper}")
            print(f"  Out of spec rate: {defect_rate:.1%}")
            print(f"  Actual defects caught: {actual_defects}/{out_of_spec.sum()}")

    print("\n2. PROCESS CONTROL ANALYSIS")
    print("-" * 40)

    # Process capability analysis
    process_features = ["machine_speed_rpm", "temperature_celsius", "pressure_bar"]

    for feature in process_features:
        normal_process = df[df["is_anomaly"] == 0][feature]
        defective_process = df[df["is_anomaly"] == 1][feature]

        print(f"{feature}:")
        print(f"  Normal: {normal_process.mean():.1f} ± {normal_process.std():.1f}")
        print(
            f"  Defective: {defective_process.mean():.1f} ± {defective_process.std():.1f}"
        )
        print(
            f"  Process variation increase: {(defective_process.std() / normal_process.std()):.2f}x"
        )

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Quality Control Integration:")
    print("     - Combine dimensional and process features")
    print("     - Use specification limits as constraints")
    print("     - Monitor process capability indices")
    print("     - Implement control charts")

    print("  2. Feature Engineering:")
    print("     - Specification deviation scores")
    print("     - Process stability indicators")
    print("     - Interaction terms between dimensions")
    print("     - Machine performance features")

    print("  3. Production Implementation:")
    print("     - Real-time quality assessment")
    print("     - Predictive maintenance integration")
    print("     - Root cause analysis features")
    print("     - Cost-based anomaly scoring")

    return df


def analyze_ecommerce_behavior():
    """Analysis approach for e-commerce behavior anomaly detection"""
    print("=" * 80)
    print("E-COMMERCE BEHAVIOR DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("ecommerce_behavior")

    print(f"Dataset: {metadata.get('description', 'E-commerce behavior analysis')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. BEHAVIOR PATTERN ANALYSIS")
    print("-" * 40)

    # Session characteristics
    normal_behavior = df[df["is_anomaly"] == 0]
    anomaly_behavior = df[df["is_anomaly"] == 1]

    behavior_metrics = [
        "session_duration_minutes",
        "pages_viewed",
        "items_clicked",
        "pages_per_minute",
    ]

    for metric in behavior_metrics:
        print(f"{metric}:")
        print(
            f"  Normal: {normal_behavior[metric].mean():.2f} ± {normal_behavior[metric].std():.2f}"
        )
        print(
            f"  Anomaly: {anomaly_behavior[metric].mean():.2f} ± {anomaly_behavior[metric].std():.2f}"
        )

    print("\n2. CONVERSION AND FRAUD ANALYSIS")
    print("-" * 40)

    # Purchase patterns
    normal_conversion = normal_behavior["made_purchase"].mean()
    anomaly_conversion = anomaly_behavior["made_purchase"].mean()

    print("Conversion rates:")
    print(f"  Normal users: {normal_conversion:.1%}")
    print(f"  Anomalous users: {anomaly_conversion:.1%}")

    # High-value transactions
    high_value_threshold = df["purchase_amount"].quantile(0.95)
    high_value_normal = (
        normal_behavior["purchase_amount"] > high_value_threshold
    ).sum()
    high_value_anomaly = (
        anomaly_behavior["purchase_amount"] > high_value_threshold
    ).sum()

    print(f"High-value transactions (>${high_value_threshold:.0f}+):")
    print(f"  Normal users: {high_value_normal}")
    print(f"  Anomalous users: {high_value_anomaly}")

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Behavior Modeling:")
    print("     - Session-based feature engineering")
    print("     - Conversion funnel analysis")
    print("     - Temporal behavior patterns")
    print("     - User journey analysis")

    print("  2. Anomaly Categories:")
    print("     - Bot detection: High speed, low engagement")
    print("     - Fraud detection: Unusual purchase patterns")
    print("     - Account takeover: Sudden behavior changes")
    print("     - Scraping detection: High page views, no interaction")

    print("  3. Real-time Implementation:")
    print("     - Streaming anomaly detection")
    print("     - Behavioral scoring systems")
    print("     - Adaptive thresholds")
    print("     - Risk-based authentication triggers")

    return df


def analyze_time_series_anomalies():
    """Analysis approach for time series anomaly detection"""
    print("=" * 80)
    print("TIME SERIES ANOMALY DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("time_series_anomalies")

    print(f"Dataset: {metadata.get('description', 'Time series anomaly detection')}")
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. TIME SERIES CHARACTERISTICS")
    print("-" * 40)

    # Basic time series stats
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    print(f"Value mean: {df['value'].mean():.2f} ± {df['value'].std():.2f}")

    # Anomaly distribution over time
    anomaly_counts = df.groupby("hour_of_day")["is_anomaly"].sum()
    peak_anomaly_hour = anomaly_counts.idxmax()
    print(
        f"Peak anomaly hour: {peak_anomaly_hour} with {anomaly_counts[peak_anomaly_hour]} anomalies"
    )

    # Moving average analysis
    ma_deviation = abs(df["value"] - df["moving_avg_5"])
    normal_deviation = ma_deviation[df["is_anomaly"] == 0].mean()
    anomaly_deviation = ma_deviation[df["is_anomaly"] == 1].mean()

    print("Deviation from 5-period moving average:")
    print(f"  Normal points: {normal_deviation:.2f}")
    print(f"  Anomaly points: {anomaly_deviation:.2f}")
    print(f"  Anomaly/Normal ratio: {anomaly_deviation / normal_deviation:.2f}x")

    print("\n2. SEASONALITY AND TREND ANALYSIS")
    print("-" * 40)

    # Daily seasonality
    hourly_mean = df.groupby("hour_of_day")["value"].mean()
    seasonality_strength = hourly_mean.std()
    print(f"Daily seasonality strength (std): {seasonality_strength:.2f}")

    # Weekly seasonality
    weekly_mean = df.groupby("day_of_week")["value"].mean()
    weekly_seasonality = weekly_mean.std()
    print(f"Weekly seasonality strength (std): {weekly_seasonality:.2f}")

    # Trend analysis using differences
    trend_strength = abs(df["first_difference"]).mean()
    print(f"Trend strength (mean absolute difference): {trend_strength:.2f}")

    print("\n3. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Time Series Preprocessing:")
    print("     - Detrending and deseasonalization")
    print("     - Outlier-robust moving averages")
    print("     - Difference features for trend detection")
    print("     - Seasonal decomposition")

    print("  2. Anomaly Detection Strategy:")
    print("     - Point anomalies: Statistical outliers")
    print("     - Contextual anomalies: Seasonal context consideration")
    print("     - Collective anomalies: Pattern change detection")
    print("     - Sliding window approaches")

    print("  3. Feature Engineering:")
    print("     - Lag features and autoregressive terms")
    print("     - Rolling statistics (mean, std, min, max)")
    print("     - Spectral features for periodicity")
    print("     - Change point detection features")

    return df


def analyze_high_dimensional():
    """Analysis approach for high-dimensional anomaly detection"""
    print("=" * 80)
    print("HIGH-DIMENSIONAL DATASET ANALYSIS")
    print("=" * 80)

    df, metadata = load_dataset_with_metadata("high_dimensional")

    print(
        f"Dataset: {metadata.get('description', 'High-dimensional anomaly detection')}"
    )
    print(
        f"Samples: {len(df):,} | Features: {len(df.columns) - 1} | Anomaly Rate: {df['is_anomaly'].mean():.1%}"
    )

    print("\n1. DIMENSIONALITY ANALYSIS")
    print("-" * 40)

    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    n_features = len(feature_cols)
    n_samples = len(df)

    print(f"Dimensionality: {n_features} features, {n_samples} samples")
    print(f"Samples-to-features ratio: {n_samples / n_features:.1f}")

    if n_samples / n_features < 10:
        print(
            "⚠️  Warning: High dimensional, low sample count - curse of dimensionality may apply"
        )

    # Feature variance analysis
    feature_vars = df[feature_cols].var()
    print(
        f"Feature variance range: {feature_vars.min():.3f} to {feature_vars.max():.3f}"
    )
    print(
        f"Feature variance mean: {feature_vars.mean():.3f} ± {feature_vars.std():.3f}"
    )

    print("\n2. CORRELATION STRUCTURE ANALYSIS")
    print("-" * 40)

    # Correlation analysis
    corr_matrix = df[feature_cols].corr()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = (abs(upper_triangle) > 0.7).sum().sum()
    moderate_corr_pairs = (
        ((abs(upper_triangle) > 0.3) & (abs(upper_triangle) <= 0.7)).sum().sum()
    )

    print(f"High correlation pairs (>0.7): {high_corr_pairs}")
    print(f"Moderate correlation pairs (0.3-0.7): {moderate_corr_pairs}")
    print(f"Total feature pairs: {len(feature_cols) * (len(feature_cols) - 1) // 2}")

    # Distance analysis
    df[df["is_anomaly"] == 0][feature_cols]
    df[df["is_anomaly"] == 1][feature_cols]

    print("\n3. DISTANCE AND DENSITY ANALYSIS")
    print("-" * 40)

    # Engineered features analysis
    engineered_features = [
        "feature_sum",
        "feature_mean",
        "feature_std",
        "feature_range",
    ]

    for feature in engineered_features:
        if feature in df.columns:
            normal_values = df[df["is_anomaly"] == 0][feature]
            anomaly_values = df[df["is_anomaly"] == 1][feature]

            print(f"{feature}:")
            print(f"  Normal: {normal_values.mean():.3f} ± {normal_values.std():.3f}")
            print(
                f"  Anomaly: {anomaly_values.mean():.3f} ± {anomaly_values.std():.3f}"
            )
            print(
                f"  Separation: {abs(normal_values.mean() - anomaly_values.mean()) / normal_values.std():.2f} σ"
            )

    print("\n4. RECOMMENDED ALGORITHMS & APPROACH")
    print("-" * 40)
    print("Best algorithms for this dataset:")
    for algo in metadata.get("recommended_algorithms", []):
        print(f"  • {algo}")

    print("\nRecommended approach:")
    print("  1. Dimensionality Considerations:")
    print("     - Use algorithms robust to curse of dimensionality")
    print("     - Consider dimensionality reduction (PCA, ICA)")
    print("     - Feature selection based on variance/correlation")
    print("     - Distance metric selection important")

    print("  2. Algorithm Selection Rationale:")
    print("     - IsolationForest: Handles high dimensions well")
    print("     - PCA-based: Leverages correlation structure")
    print("     - ABOD: Angle-based, less affected by dimensionality")
    print("     - LOF: May struggle in very high dimensions")

    print("  3. Feature Engineering:")
    print("     - Aggregate features (sums, means, variances)")
    print("     - Principal component features")
    print("     - Distance-based features")
    print("     - Density-based features")

    print("  4. Validation Strategy:")
    print("     - Cross-validation with stratification")
    print("     - Multiple random splits")
    print("     - Stability analysis across runs")
    print("     - Parameter sensitivity analysis")

    return df


def main():
    """Run comprehensive analysis for all datasets"""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS FOR PYNOMALY")
    print("=" * 80)
    print("This script provides detailed analysis and recommendations")
    print("for each type of dataset in the sample collection.")
    print("=" * 80)

    # Analysis functions for each dataset type
    analysis_functions = [
        analyze_financial_fraud,
        analyze_network_intrusion,
        analyze_iot_sensors,
        analyze_manufacturing_quality,
        analyze_ecommerce_behavior,
        analyze_time_series_anomalies,
        analyze_high_dimensional,
    ]

    results = {}

    for analysis_func in analysis_functions:
        try:
            print("\n" * 2)
            df = analysis_func()
            results[analysis_func.__name__] = {
                "status": "success",
                "samples": len(df),
                "features": len(df.columns) - 1,
                "anomaly_rate": df["is_anomaly"].mean(),
            }
        except Exception as e:
            print(f"❌ Error in {analysis_func.__name__}: {e}")
            results[analysis_func.__name__] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" * 2)
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)

    print(f"Completed: {successful}/{total} dataset analyses")

    if successful > 0:
        print("\nDataset Overview:")
        for name, result in results.items():
            if result["status"] == "success":
                dataset_name = name.replace("analyze_", "").replace("_", " ").title()
                print(
                    f"  • {dataset_name}: {result['samples']:,} samples, {result['features']} features, {result['anomaly_rate']:.1%} anomalies"
                )

    print("\n📋 For detailed documentation, see:")
    print("   📖 examples/sample_datasets/README.md")
    print("   📁 examples/sample_datasets/")
    print("   🧪 This analysis script: scripts/analyze_dataset_comprehensive.py")


if __name__ == "__main__":
    main()
