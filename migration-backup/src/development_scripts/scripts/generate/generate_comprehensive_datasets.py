#!/usr/bin/env python3
"""
Comprehensive Dataset Generator for Pynomaly

Generates a wide variety of tabular datasets covering different scenarios,
data distributions, anomaly types, and outlier patterns for testing
and demonstration purposes.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure reproducibility
np.random.seed(42)


def create_financial_fraud_dataset(n_samples=10000, anomaly_rate=0.02):
    """
    Financial transaction fraud dataset
    Anomalies: Unusual transaction amounts, patterns, timing
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal transactions
    normal_amounts = np.random.lognormal(mean=4, sigma=1, size=n_normal)  # ~$55 average
    normal_hours = np.random.choice(
        range(6, 23),
        size=n_normal,
        p=[
            0.05,
            0.08,
            0.12,
            0.15,
            0.15,
            0.12,
            0.08,
            0.05,
            0.04,
            0.03,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.01,
        ],
    )
    normal_merchant_categories = np.random.choice(
        ["grocery", "gas", "restaurant", "retail", "online"],
        size=n_normal,
        p=[0.3, 0.2, 0.25, 0.15, 0.1],
    )
    normal_locations = np.random.choice(
        range(1, 51), size=n_normal
    )  # 50 different locations
    normal_freq = np.random.poisson(lam=3, size=n_normal)  # transactions per day

    # Fraudulent transactions - unusual patterns
    fraud_amounts = np.concatenate(
        [
            np.random.uniform(5000, 15000, size=n_anomalies // 3),  # Large amounts
            np.random.uniform(0.01, 1, size=n_anomalies // 3),  # Micro amounts
            np.random.lognormal(
                mean=6, sigma=2, size=n_anomalies - 2 * (n_anomalies // 3)
            ),  # Very high amounts
        ]
    )
    fraud_hours = np.random.choice([1, 2, 3, 4, 5], size=n_anomalies)  # Unusual hours
    fraud_merchant_categories = np.random.choice(
        ["atm", "casino", "luxury", "crypto", "unknown"], size=n_anomalies
    )
    fraud_locations = np.random.choice(
        range(51, 101), size=n_anomalies
    )  # Different locations
    fraud_freq = np.random.choice(
        [0, 15, 20, 25], size=n_anomalies
    )  # Unusual frequency

    # Combine data
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    hours = np.concatenate([normal_hours, fraud_hours])
    categories = np.concatenate([normal_merchant_categories, fraud_merchant_categories])
    locations = np.concatenate([normal_locations, fraud_locations])
    frequencies = np.concatenate([normal_freq, fraud_freq])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Additional engineered features
    is_weekend = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    velocity_score = amounts / (frequencies + 1)  # Amount per transaction frequency

    # Create categorical encodings
    category_encoded = pd.Categorical(categories).codes

    df = pd.DataFrame(
        {
            "transaction_amount": amounts,
            "hour_of_day": hours,
            "merchant_category": categories,
            "merchant_category_encoded": category_encoded,
            "location_id": locations,
            "daily_frequency": frequencies,
            "is_weekend": is_weekend,
            "velocity_score": velocity_score,
            "amount_log": np.log1p(amounts),
            "is_anomaly": labels.astype(int),
        }
    )

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def create_network_intrusion_dataset(n_samples=8000, anomaly_rate=0.05):
    """
    Network intrusion detection dataset
    Anomalies: DDoS attacks, port scanning, unusual traffic patterns
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal network traffic
    normal_packets = np.random.gamma(shape=2, scale=50, size=n_normal)
    normal_bytes = normal_packets * np.random.gamma(shape=3, scale=200, size=n_normal)
    normal_duration = np.random.exponential(scale=10, size=n_normal)
    normal_ports = np.random.choice(
        [80, 443, 22, 21, 25, 53], size=n_normal, p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.1]
    )
    normal_protocols = np.random.choice(
        ["TCP", "UDP", "ICMP"], size=n_normal, p=[0.7, 0.25, 0.05]
    )

    # Malicious traffic patterns
    attack_packets = np.concatenate(
        [
            np.random.gamma(
                shape=10, scale=500, size=n_anomalies // 3
            ),  # DDoS - high packet count
            np.random.uniform(
                1, 5, size=n_anomalies // 3
            ),  # Port scanning - low packets
            np.random.gamma(
                shape=5, scale=100, size=n_anomalies - 2 * (n_anomalies // 3)
            ),
        ]
    )
    attack_bytes = attack_packets * np.random.gamma(
        shape=1, scale=50, size=n_anomalies
    )  # Different size pattern
    attack_duration = np.concatenate(
        [
            np.random.uniform(0.1, 1, size=n_anomalies // 2),  # Very short
            np.random.uniform(
                100, 500, size=n_anomalies - n_anomalies // 2
            ),  # Very long
        ]
    )
    attack_ports = np.random.choice(
        range(1024, 65536), size=n_anomalies
    )  # Random high ports
    attack_protocols = np.random.choice(
        ["TCP", "UDP", "ICMP"], size=n_anomalies, p=[0.5, 0.4, 0.1]
    )

    # Combine
    packets = np.concatenate([normal_packets, attack_packets])
    bytes_transferred = np.concatenate([normal_bytes, attack_bytes])
    duration = np.concatenate([normal_duration, attack_duration])
    ports = np.concatenate([normal_ports, attack_ports])
    protocols = np.concatenate([normal_protocols, attack_protocols])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Additional features
    packets_per_second = packets / (duration + 0.001)
    bytes_per_packet = bytes_transferred / (packets + 1)
    protocol_encoded = pd.Categorical(protocols).codes
    port_is_standard = (ports <= 1024).astype(int)

    df = pd.DataFrame(
        {
            "packet_count": packets,
            "bytes_transferred": bytes_transferred,
            "duration_seconds": duration,
            "destination_port": ports,
            "protocol": protocols,
            "protocol_encoded": protocol_encoded,
            "packets_per_second": packets_per_second,
            "bytes_per_packet": bytes_per_packet,
            "port_is_standard": port_is_standard,
            "log_packets": np.log1p(packets),
            "log_bytes": np.log1p(bytes_transferred),
            "is_anomaly": labels.astype(int),
        }
    )

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_iot_sensor_dataset(n_samples=12000, anomaly_rate=0.03):
    """
    IoT sensor monitoring dataset
    Anomalies: Sensor failures, environmental anomalies, drift
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal sensor readings (temperature, humidity, pressure, vibration)
    time_hours = np.linspace(0, 24 * 30, n_normal)  # 30 days
    temp_base = 20 + 5 * np.sin(2 * np.pi * time_hours / 24)  # Daily cycle
    temp_normal = temp_base + np.random.normal(0, 1, n_normal)

    humidity_base = 45 + 10 * np.sin(2 * np.pi * time_hours / 24 + np.pi / 4)
    humidity_normal = humidity_base + np.random.normal(0, 2, n_normal)

    pressure_normal = np.random.normal(1013, 5, n_normal)  # Atmospheric pressure
    vibration_normal = np.random.gamma(shape=2, scale=0.5, size=n_normal)

    # Anomalous sensor readings
    time_anomaly = np.random.uniform(0, 24 * 30, n_anomalies)
    temp_anomaly = np.concatenate(
        [
            np.random.uniform(
                -10, 5, size=n_anomalies // 3
            ),  # Sensor failure - too cold
            np.random.uniform(35, 60, size=n_anomalies // 3),  # Overheating
            np.random.normal(
                25, 15, size=n_anomalies - 2 * (n_anomalies // 3)
            ),  # High variance
        ]
    )

    humidity_anomaly = np.concatenate(
        [
            np.random.uniform(0, 10, size=n_anomalies // 2),  # Too dry
            np.random.uniform(
                90, 100, size=n_anomalies - n_anomalies // 2
            ),  # Too humid
        ]
    )

    pressure_anomaly = np.random.normal(1013, 20, n_anomalies)  # High variance
    vibration_anomaly = np.random.gamma(
        shape=10, scale=2, size=n_anomalies
    )  # High vibration

    # Combine
    time_all = np.concatenate([time_hours, time_anomaly])
    temperature = np.concatenate([temp_normal, temp_anomaly])
    humidity = np.concatenate([humidity_normal, humidity_anomaly])
    pressure = np.concatenate([pressure_normal, pressure_anomaly])
    vibration = np.concatenate([vibration_normal, vibration_anomaly])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Additional features
    temp_humidity_ratio = temperature / (humidity + 1)
    pressure_deviation = np.abs(pressure - 1013)
    vibration_log = np.log1p(vibration)
    hour_of_day = (time_all % 24).astype(int)
    day_of_month = (time_all // 24).astype(int) + 1

    df = pd.DataFrame(
        {
            "timestamp_hours": time_all,
            "temperature_celsius": temperature,
            "humidity_percent": humidity,
            "pressure_hpa": pressure,
            "vibration_level": vibration,
            "temp_humidity_ratio": temp_humidity_ratio,
            "pressure_deviation": pressure_deviation,
            "vibration_log": vibration_log,
            "hour_of_day": hour_of_day,
            "day_of_month": day_of_month,
            "is_anomaly": labels.astype(int),
        }
    )

    # Sort by time
    df = df.sort_values("timestamp_hours").reset_index(drop=True)
    return df


def create_manufacturing_quality_dataset(n_samples=6000, anomaly_rate=0.08):
    """
    Manufacturing quality control dataset
    Anomalies: Defective products, machine malfunctions, process variations
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal production measurements
    dimension_1 = np.random.normal(100, 2, n_normal)  # Target: 100mm ±2mm
    dimension_2 = np.random.normal(50, 1, n_normal)  # Target: 50mm ±1mm
    weight = np.random.normal(500, 10, n_normal)  # Target: 500g ±10g
    surface_roughness = np.random.gamma(shape=2, scale=1.5, size=n_normal)
    hardness = np.random.normal(60, 3, n_normal)  # HRC scale

    # Machine settings for normal production
    machine_speed = np.random.normal(1000, 50, n_normal)  # RPM
    temperature = np.random.normal(180, 5, n_normal)  # °C
    pressure = np.random.normal(50, 2, n_normal)  # Bar

    # Defective products - out of specification
    defect_dim1 = np.concatenate(
        [
            np.random.normal(85, 3, n_anomalies // 3),  # Undersized
            np.random.normal(115, 4, n_anomalies // 3),  # Oversized
            np.random.uniform(
                70, 130, n_anomalies - 2 * (n_anomalies // 3)
            ),  # Random defects
        ]
    )

    defect_dim2 = np.concatenate(
        [
            np.random.normal(45, 2, n_anomalies // 2),
            np.random.normal(58, 3, n_anomalies - n_anomalies // 2),
        ]
    )

    defect_weight = np.random.normal(500, 30, n_anomalies)  # High variance
    defect_roughness = np.random.gamma(shape=5, scale=3, size=n_anomalies)  # Too rough
    defect_hardness = np.concatenate(
        [
            np.random.normal(40, 5, n_anomalies // 2),  # Too soft
            np.random.normal(80, 8, n_anomalies - n_anomalies // 2),  # Too hard
        ]
    )

    # Machine settings during defects
    defect_speed = np.random.normal(1000, 150, n_anomalies)  # High variance
    defect_temp = np.random.normal(180, 20, n_anomalies)  # High variance
    defect_pressure = np.random.normal(50, 8, n_anomalies)  # High variance

    # Combine
    all_dim1 = np.concatenate([dimension_1, defect_dim1])
    all_dim2 = np.concatenate([dimension_2, defect_dim2])
    all_weight = np.concatenate([weight, defect_weight])
    all_roughness = np.concatenate([surface_roughness, defect_roughness])
    all_hardness = np.concatenate([hardness, defect_hardness])
    all_speed = np.concatenate([machine_speed, defect_speed])
    all_temp = np.concatenate([temperature, defect_temp])
    all_pressure = np.concatenate([pressure, defect_pressure])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Additional features
    dimension_ratio = all_dim1 / all_dim2
    weight_per_volume = all_weight / (all_dim1 * all_dim2 * 10)  # Simplified volume
    spec_deviation = np.sqrt(
        (all_dim1 - 100) ** 2 + (all_dim2 - 50) ** 2 + ((all_weight - 500) / 10) ** 2
    )

    df = pd.DataFrame(
        {
            "dimension_1_mm": all_dim1,
            "dimension_2_mm": all_dim2,
            "weight_grams": all_weight,
            "surface_roughness": all_roughness,
            "hardness_hrc": all_hardness,
            "machine_speed_rpm": all_speed,
            "temperature_celsius": all_temp,
            "pressure_bar": all_pressure,
            "dimension_ratio": dimension_ratio,
            "weight_per_volume": weight_per_volume,
            "specification_deviation": spec_deviation,
            "is_anomaly": labels.astype(int),
        }
    )

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_ecommerce_behavior_dataset(n_samples=15000, anomaly_rate=0.04):
    """
    E-commerce user behavior dataset
    Anomalies: Bot behavior, fraud, unusual purchase patterns
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal user behavior
    session_duration = np.random.lognormal(mean=3, sigma=1, size=n_normal)  # Minutes
    pages_viewed = np.random.poisson(lam=8, size=n_normal)
    items_clicked = np.random.poisson(lam=5, size=n_normal)
    cart_additions = np.random.poisson(lam=2, size=n_normal)
    purchases = np.random.binomial(n=1, p=0.15, size=n_normal)

    # Normal purchase amounts (for those who purchase)
    purchase_amounts = np.where(
        purchases == 1, np.random.lognormal(mean=4, sigma=0.8, size=n_normal), 0
    )

    # Time features (normal users) - normalized probabilities
    hour_probs = np.array(
        [
            0.02,
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,
            0.03,
            0.05,
            0.06,
            0.07,
            0.08,
            0.08,
            0.08,
            0.08,
            0.07,
            0.06,
            0.05,
            0.04,
            0.04,
            0.04,
            0.04,
            0.03,
            0.03,
            0.02,
        ]
    )
    hour_probs = hour_probs / hour_probs.sum()  # Normalize to sum to 1
    hours = np.random.choice(range(24), size=n_normal, p=hour_probs)

    # Anomalous behavior patterns
    # Bot-like behavior: Very fast, many pages, no purchases
    bot_duration = np.random.uniform(0.1, 2, size=n_anomalies // 3)
    bot_pages = np.random.poisson(lam=50, size=n_anomalies // 3)
    bot_clicks = np.random.poisson(lam=100, size=n_anomalies // 3)
    bot_cart = np.zeros(n_anomalies // 3)
    bot_purchases = np.zeros(n_anomalies // 3)
    bot_amounts = np.zeros(n_anomalies // 3)
    bot_hours = np.random.choice([2, 3, 4, 5], size=n_anomalies // 3)

    # Fraud-like: Quick high-value purchases
    fraud_duration = np.random.uniform(1, 5, size=n_anomalies // 3)
    fraud_pages = np.random.poisson(lam=3, size=n_anomalies // 3)
    fraud_clicks = np.random.poisson(lam=2, size=n_anomalies // 3)
    fraud_cart = np.random.poisson(lam=1, size=n_anomalies // 3)
    fraud_purchases = np.ones(n_anomalies // 3)
    fraud_amounts = np.random.uniform(1000, 5000, size=n_anomalies // 3)
    fraud_hours = np.random.choice(range(24), size=n_anomalies // 3)

    # Unusual browsing: Long sessions, no purchases
    browse_duration = np.random.uniform(
        60, 300, size=n_anomalies - 2 * (n_anomalies // 3)
    )
    browse_pages = np.random.poisson(lam=200, size=n_anomalies - 2 * (n_anomalies // 3))
    browse_clicks = np.random.poisson(lam=50, size=n_anomalies - 2 * (n_anomalies // 3))
    browse_cart = np.random.poisson(lam=10, size=n_anomalies - 2 * (n_anomalies // 3))
    browse_purchases = np.zeros(n_anomalies - 2 * (n_anomalies // 3))
    browse_amounts = np.zeros(n_anomalies - 2 * (n_anomalies // 3))
    browse_hours = np.random.choice(
        range(24), size=n_anomalies - 2 * (n_anomalies // 3)
    )

    # Combine anomalous data
    anom_duration = np.concatenate([bot_duration, fraud_duration, browse_duration])
    anom_pages = np.concatenate([bot_pages, fraud_pages, browse_pages])
    anom_clicks = np.concatenate([bot_clicks, fraud_clicks, browse_clicks])
    anom_cart = np.concatenate([bot_cart, fraud_cart, browse_cart])
    anom_purchases = np.concatenate([bot_purchases, fraud_purchases, browse_purchases])
    anom_amounts = np.concatenate([bot_amounts, fraud_amounts, browse_amounts])
    anom_hours = np.concatenate([bot_hours, fraud_hours, browse_hours])

    # Combine all data
    all_duration = np.concatenate([session_duration, anom_duration])
    all_pages = np.concatenate([pages_viewed, anom_pages])
    all_clicks = np.concatenate([items_clicked, anom_clicks])
    all_cart = np.concatenate([cart_additions, anom_cart])
    all_purchases = np.concatenate([purchases, anom_purchases])
    all_amounts = np.concatenate([purchase_amounts, anom_amounts])
    all_hours = np.concatenate([hours, anom_hours])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Additional features
    click_rate = all_clicks / (all_pages + 1)
    cart_rate = all_cart / (all_pages + 1)
    conversion_rate = all_purchases / (all_pages + 1)
    pages_per_minute = all_pages / (all_duration + 0.1)
    amount_per_page = all_amounts / (all_pages + 1)

    df = pd.DataFrame(
        {
            "session_duration_minutes": all_duration,
            "pages_viewed": all_pages,
            "items_clicked": all_clicks,
            "cart_additions": all_cart,
            "made_purchase": all_purchases,
            "purchase_amount": all_amounts,
            "hour_of_day": all_hours,
            "click_through_rate": click_rate,
            "cart_conversion_rate": cart_rate,
            "purchase_conversion_rate": conversion_rate,
            "pages_per_minute": pages_per_minute,
            "amount_per_page_viewed": amount_per_page,
            "is_anomaly": labels.astype(int),
        }
    )

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_time_series_anomaly_dataset(n_samples=5000, anomaly_rate=0.06):
    """
    Time series with various anomaly patterns
    Anomalies: Spikes, drops, trend changes, seasonality breaks
    """
    n_anomalies = int(n_samples * anomaly_rate)

    # Generate time series
    time = np.arange(n_samples)

    # Base trend and seasonality
    trend = 0.001 * time + 10
    daily_season = 2 * np.sin(2 * np.pi * time / 24)
    weekly_season = 1.5 * np.sin(2 * np.pi * time / (24 * 7))
    noise = np.random.normal(0, 0.5, n_samples)

    # Normal time series
    normal_ts = trend + daily_season + weekly_season + noise

    # Add various types of anomalies
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    anomaly_ts = normal_ts.copy()

    for i, idx in enumerate(anomaly_indices):
        anomaly_type = i % 4
        if anomaly_type == 0:  # Spike
            anomaly_ts[idx] += np.random.uniform(5, 15)
        elif anomaly_type == 1:  # Drop
            anomaly_ts[idx] -= np.random.uniform(3, 10)
        elif anomaly_type == 2:  # Level shift
            shift_length = min(20, n_samples - idx)
            anomaly_ts[idx : idx + shift_length] += np.random.uniform(2, 8)
        else:  # Trend change
            change_length = min(30, n_samples - idx)
            trend_change = np.linspace(0, np.random.uniform(3, 10), change_length)
            anomaly_ts[idx : idx + change_length] += trend_change

    # Additional features
    moving_avg_5 = (
        pd.Series(anomaly_ts)
        .rolling(window=5, center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    moving_std_5 = (
        pd.Series(anomaly_ts).rolling(window=5, center=True).std().fillna(0.5)
    )
    diff_1 = np.diff(anomaly_ts, prepend=anomaly_ts[0])
    diff_2 = np.diff(diff_1, prepend=diff_1[0])

    # Create anomaly labels
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1

    df = pd.DataFrame(
        {
            "timestamp": time,
            "value": anomaly_ts,
            "moving_avg_5": moving_avg_5,
            "moving_std_5": moving_std_5,
            "first_difference": diff_1,
            "second_difference": diff_2,
            "hour_of_day": time % 24,
            "day_of_week": (time // 24) % 7,
            "value_vs_ma_ratio": anomaly_ts / (moving_avg_5 + 0.001),
            "std_normalized_value": (anomaly_ts - moving_avg_5)
            / (moving_std_5 + 0.001),
            "is_anomaly": labels.astype(int),
        }
    )

    return df


def create_high_dimensional_dataset(n_samples=3000, n_features=50, anomaly_rate=0.1):
    """
    High-dimensional dataset with correlated features
    Anomalies: Outliers in high-dimensional space
    """
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Create correlation structure
    base_features = 5
    correlation_matrix = np.random.uniform(0.3, 0.8, (base_features, base_features))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    # Generate base correlated features
    base_data = np.random.multivariate_normal(
        mean=np.zeros(base_features), cov=correlation_matrix, size=n_normal
    )

    # Generate additional features as combinations of base features
    additional_normal = []
    for _i in range(n_features - base_features):
        # Linear combinations with noise
        weights = np.random.uniform(-1, 1, base_features)
        feature = np.dot(base_data, weights) + np.random.normal(0, 0.5, n_normal)
        additional_normal.append(feature)

    normal_data = np.column_stack([base_data] + additional_normal)

    # Generate anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.random.uniform(-3, 3, n_features),
        cov=np.eye(n_features) * np.random.uniform(2, 5),
        size=n_anomalies,
    )

    # Combine
    all_data = np.vstack([normal_data, anomaly_data])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Create DataFrame
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(all_data, columns=feature_names)
    df["is_anomaly"] = labels.astype(int)

    # Add some engineered features
    df["feature_sum"] = df[feature_names].sum(axis=1)
    df["feature_mean"] = df[feature_names].mean(axis=1)
    df["feature_std"] = df[feature_names].std(axis=1)
    df["feature_range"] = df[feature_names].max(axis=1) - df[feature_names].min(axis=1)

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def download_real_world_datasets():
    """
    Download and prepare real-world datasets
    """
    datasets_info = []

    try:
        from sklearn.datasets import fetch_kddcup99

        # KDD Cup 1999 - Network intrusion detection
        print("Downloading KDD Cup 1999 dataset...")
        kdd_data = fetch_kddcup99(subset="SA", percent10=True, return_X_y=True)
        X, y = kdd_data

        # Convert to DataFrame
        feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
        df_kdd = pd.DataFrame(X, columns=feature_names)
        df_kdd["is_anomaly"] = (y != b"normal.").astype(int)

        # Take a sample to make it manageable
        df_kdd_sample = df_kdd.sample(n=10000, random_state=42)

        datasets_info.append(
            {
                "name": "kdd_cup_1999",
                "description": "Network intrusion detection dataset from KDD Cup 1999",
                "source": "UCI Machine Learning Repository",
                "anomaly_rate": df_kdd_sample["is_anomaly"].mean(),
                "samples": len(df_kdd_sample),
                "features": X.shape[1],
            }
        )

        return [("kdd_cup_1999", df_kdd_sample)], datasets_info

    except Exception as e:
        print(f"Error downloading real-world datasets: {e}")
        return [], []


def create_metadata(
    dataset_name, df, description, anomaly_types, recommended_algorithms
):
    """Create metadata for a dataset"""
    return {
        "name": dataset_name,
        "description": description,
        "samples": int(len(df)),
        "features": int(len(df.columns) - 1),  # Excluding label column
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "anomaly_count": int(df["is_anomaly"].sum()),
        "anomaly_types": anomaly_types,
        "recommended_algorithms": recommended_algorithms,
        "feature_types": {
            "numerical": int(len(df.select_dtypes(include=[np.number]).columns) - 1),
            "categorical": int(
                len(df.select_dtypes(include=[object, "category"]).columns)
            ),
        },
        "missing_values": int(df.isnull().sum().sum()),
        "created": datetime.now().isoformat(),
        "file_size_mb": float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)),
    }


def main():
    """Generate all datasets"""
    print("🔄 Generating comprehensive tabular datasets for Pynomaly...")

    # Create output directories
    base_dir = Path("examples/sample_datasets")
    synthetic_dir = base_dir / "synthetic"
    real_world_dir = base_dir / "real_world"

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    real_world_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    # Generate synthetic datasets
    synthetic_datasets = [
        (
            "financial_fraud",
            create_financial_fraud_dataset(),
            "Financial transaction fraud detection with unusual amounts, timing, and patterns",
            [
                "Transaction fraud",
                "Unusual amounts",
                "Timing anomalies",
                "Location anomalies",
            ],
            ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
        ),
        (
            "network_intrusion",
            create_network_intrusion_dataset(),
            "Network traffic anomaly detection with DDoS, port scanning, and malicious patterns",
            [
                "DDoS attacks",
                "Port scanning",
                "Traffic volume anomalies",
                "Protocol anomalies",
            ],
            ["IsolationForest", "EllipticEnvelope", "PyOD.ABOD"],
        ),
        (
            "iot_sensors",
            create_iot_sensor_dataset(),
            "IoT sensor monitoring with failures, environmental anomalies, and drift",
            [
                "Sensor failures",
                "Environmental anomalies",
                "Measurement drift",
                "Temporal anomalies",
            ],
            ["LocalOutlierFactor", "EllipticEnvelope", "PyOD.KNN"],
        ),
        (
            "manufacturing_quality",
            create_manufacturing_quality_dataset(),
            "Manufacturing quality control with defective products and process variations",
            [
                "Product defects",
                "Process variations",
                "Machine malfunctions",
                "Specification violations",
            ],
            ["IsolationForest", "PyOD.OCSVM", "EllipticEnvelope"],
        ),
        (
            "ecommerce_behavior",
            create_ecommerce_behavior_dataset(),
            "E-commerce user behavior with bot detection and fraud patterns",
            [
                "Bot behavior",
                "Purchase fraud",
                "Unusual browsing patterns",
                "Session anomalies",
            ],
            ["LocalOutlierFactor", "IsolationForest", "PyOD.COPOD"],
        ),
        (
            "time_series_anomalies",
            create_time_series_anomaly_dataset(),
            "Time series with spikes, drops, trend changes, and seasonality breaks",
            [
                "Value spikes",
                "Value drops",
                "Trend changes",
                "Level shifts",
                "Seasonality breaks",
            ],
            ["PyOD.KNN", "LocalOutlierFactor", "EllipticEnvelope"],
        ),
        (
            "high_dimensional",
            create_high_dimensional_dataset(),
            "High-dimensional dataset with correlated features and outliers",
            [
                "High-dimensional outliers",
                "Correlation anomalies",
                "Feature space anomalies",
            ],
            ["IsolationForest", "PyOD.PCA", "PyOD.ABOD", "LocalOutlierFactor"],
        ),
    ]

    # Save synthetic datasets
    for (
        name,
        df,
        description,
        anomaly_types,
        recommended_algorithms,
    ) in synthetic_datasets:
        print(f"  📊 Saving {name}...")

        # Save data
        file_path = synthetic_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)

        # Create metadata
        metadata = create_metadata(
            name, df, description, anomaly_types, recommended_algorithms
        )
        all_metadata.append(metadata)

        # Save individual metadata
        with open(synthetic_dir / f"{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"    ✓ {len(df)} samples, {len(df.columns) - 1} features, {df['is_anomaly'].mean():.1%} anomalies"
        )

    # Download real-world datasets
    print("\n🌐 Downloading real-world datasets...")
    real_datasets, real_metadata = download_real_world_datasets()

    for name, df in real_datasets:
        print(f"  📊 Saving {name}...")
        file_path = real_world_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"    ✓ {len(df)} samples, {len(df.columns) - 1} features")

    all_metadata.extend(real_metadata)

    # Save master metadata
    master_metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_datasets": len(all_metadata),
        "synthetic_datasets": len(synthetic_datasets),
        "real_world_datasets": len(real_datasets),
        "datasets": all_metadata,
    }

    with open(base_dir / "datasets_metadata.json", "w") as f:
        json.dump(master_metadata, f, indent=2)

    # Create summary README
    readme_content = f"""# Pynomaly Sample Datasets

This directory contains a comprehensive collection of tabular datasets for testing and demonstrating Pynomaly's anomaly detection capabilities.

## Dataset Overview

- **Total Datasets**: {len(all_metadata)}
- **Synthetic Datasets**: {len(synthetic_datasets)}
- **Real-world Datasets**: {len(real_datasets)}
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Synthetic Datasets

"""

    for metadata in all_metadata:
        if metadata["name"] not in [rd[0] for rd in real_datasets]:
            readme_content += f"""### {metadata["name"].replace("_", " ").title()}
- **File**: `synthetic/{metadata["name"]}.csv`
- **Description**: {metadata["description"]}
- **Samples**: {metadata["samples"]:,}
- **Features**: {metadata["features"]}
- **Anomaly Rate**: {metadata["anomaly_rate"]:.1%}
- **Anomaly Types**: {", ".join(metadata["anomaly_types"])}
- **Recommended Algorithms**: {", ".join(metadata["recommended_algorithms"])}

"""

    if real_datasets:
        readme_content += "\n## Real-world Datasets\n\n"
        for metadata in all_metadata:
            if metadata["name"] in [rd[0] for rd in real_datasets]:
                readme_content += f"""### {metadata["name"].replace("_", " ").title()}
- **File**: `real_world/{metadata["name"]}.csv`
- **Description**: {metadata["description"]}
- **Samples**: {metadata["samples"]:,}
- **Features**: {metadata["features"]}
- **Anomaly Rate**: {metadata["anomaly_rate"]:.1%}

"""

    readme_content += """
## Usage Examples

See the `scripts/` directory for analysis examples and the `docs/` directory for detailed guides on how to analyze each dataset type.

## Dataset Characteristics

Each dataset is designed to test different aspects of anomaly detection:

1. **Financial Fraud**: Tests detection of transaction anomalies
2. **Network Intrusion**: Tests traffic pattern anomalies
3. **IoT Sensors**: Tests time-series and environmental anomalies
4. **Manufacturing Quality**: Tests process control anomalies
5. **E-commerce Behavior**: Tests behavioral pattern anomalies
6. **Time Series**: Tests temporal anomalies and trend changes
7. **High Dimensional**: Tests curse of dimensionality handling

## File Format

All datasets are saved as CSV files with:
- One row per sample
- Features as columns
- `is_anomaly` column (0 = normal, 1 = anomaly)
- Consistent naming conventions
"""

    with open(base_dir / "README.md", "w") as f:
        f.write(readme_content)

    print("\n✅ Dataset generation complete!")
    print(f"   📁 Saved to: {base_dir}")
    print(f"   📊 Total datasets: {len(all_metadata)}")
    print(f"   📈 Total samples: {sum(m['samples'] for m in all_metadata):,}")
    print(f"   📋 See {base_dir}/README.md for details")


if __name__ == "__main__":
    main()
