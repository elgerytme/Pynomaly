#!/usr/bin/env python3
"""
Example Dataset Generator for Anomaly Detection Tutorials
========================================================

This script generates various types of example datasets commonly used in 
anomaly detection tutorials and documentation. All datasets are saved as 
CSV files and can be used with the quickstart templates and examples.

Usage:
    python generate_example_data.py

Generated datasets:
    - credit_card_transactions.csv: Financial fraud detection
    - network_traffic.csv: Network intrusion detection  
    - sensor_readings.csv: IoT device monitoring
    - server_metrics.csv: IT infrastructure monitoring
    - manufacturing_quality.csv: Quality control monitoring
    - user_behavior.csv: User activity analysis
    - time_series_anomalies.csv: Temporal anomaly detection
    - mixed_features.csv: General purpose anomaly detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def generate_credit_card_data(n_samples=10000, anomaly_rate=0.02):
    """Generate credit card transaction data with fraud cases."""
    np.random.seed(42)
    
    # Normal transactions
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Transaction amounts (log-normal distribution)
    normal_amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=normal_count)
    normal_amounts = np.clip(normal_amounts, 1, 5000)  # $1 to $5000
    
    # Transaction times (business hours weighted)
    hour_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.08, 0.12, 0.15,
                          0.15, 0.12, 0.08, 0.08, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01,
                          0.01, 0.01, 0.01, 0.01])
    hour_probs = hour_probs / hour_probs.sum()  # Normalize to ensure sum = 1
    normal_hours = np.random.choice(range(24), normal_count, p=hour_probs)
    
    # Merchant categories (weighted by frequency)
    normal_merchants = np.random.choice(
        [1, 2, 3, 4, 5], normal_count,
        p=[0.4, 0.25, 0.2, 0.1, 0.05]  # Grocery, Gas, Restaurant, Retail, Other
    )
    
    # Days since last transaction
    normal_days_since = np.random.exponential(scale=2.0, size=normal_count)
    normal_days_since = np.clip(normal_days_since, 0.1, 30)
    
    # Location risk (low for normal transactions)
    normal_location_risk = np.random.beta(a=2, b=8, size=normal_count)
    
    # Create normal transaction DataFrame
    normal_df = pd.DataFrame({
        'transaction_id': range(normal_count),
        'amount': normal_amounts,
        'hour': normal_hours,
        'merchant_category': normal_merchants,
        'days_since_last': normal_days_since,
        'location_risk': normal_location_risk,
        'is_fraud': False
    })
    
    # Fraudulent transactions
    fraud_count = n_samples - normal_count
    
    # Fraud patterns: high amounts, unusual times, risky locations
    fraud_amounts = np.random.lognormal(mean=6.0, sigma=1.0, size=fraud_count)
    fraud_amounts = np.clip(fraud_amounts, 100, 50000)  # $100 to $50,000
    
    fraud_hours = np.random.choice(range(24), fraud_count)  # Any time
    fraud_merchants = np.random.choice([1, 2, 3, 4, 5], fraud_count)
    fraud_days_since = np.random.exponential(scale=0.1, size=fraud_count)  # Rapid succession
    fraud_location_risk = np.random.beta(a=8, b=2, size=fraud_count)  # High risk
    
    fraud_df = pd.DataFrame({
        'transaction_id': range(normal_count, n_samples),
        'amount': fraud_amounts,
        'hour': fraud_hours,
        'merchant_category': fraud_merchants,
        'days_since_last': fraud_days_since,
        'location_risk': fraud_location_risk,
        'is_fraud': True
    })
    
    # Combine and shuffle
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def generate_network_traffic_data(n_samples=20000, anomaly_rate=0.05):
    """Generate network traffic data with intrusion attempts."""
    np.random.seed(43)
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal traffic patterns
    normal_packet_size = np.random.normal(512, 128, normal_count)
    normal_packet_size = np.clip(normal_packet_size, 64, 1500)
    
    normal_duration = np.random.exponential(scale=5.0, size=normal_count)
    normal_duration = np.clip(normal_duration, 0.1, 60)
    
    normal_src_bytes = np.random.lognormal(mean=8, sigma=2, size=normal_count)
    normal_dst_bytes = np.random.lognormal(mean=7, sigma=1.5, size=normal_count)
    
    normal_protocol = np.random.choice([1, 2, 3], normal_count, p=[0.6, 0.3, 0.1])  # TCP, UDP, ICMP
    normal_flag_count = np.random.poisson(lam=2, size=normal_count)
    
    normal_df = pd.DataFrame({
        'duration': normal_duration,
        'protocol_type': normal_protocol,
        'src_bytes': normal_src_bytes,
        'dst_bytes': normal_dst_bytes,
        'flag_count': normal_flag_count,
        'packet_size_avg': normal_packet_size,
        'is_attack': False
    })
    
    # Attack traffic (anomalous patterns)
    attack_count = n_samples - normal_count
    
    # Attacks often have extreme values
    third = attack_count // 3
    remainder = attack_count - 2 * third
    attack_packet_size = np.concatenate([
        np.random.normal(64, 10, third),              # Small packets (DoS)
        np.random.normal(1400, 50, third),            # Large packets (DDoS)
        np.random.normal(512, 200, remainder)         # Variable size
    ])
    
    half = attack_count // 2
    remainder = attack_count - half
    attack_duration = np.concatenate([
        np.random.exponential(scale=0.1, size=half),     # Very short
        np.random.exponential(scale=30, size=remainder)  # Very long
    ])
    
    attack_src_bytes = np.random.lognormal(mean=10, sigma=3, size=attack_count)
    attack_dst_bytes = np.random.lognormal(mean=5, sigma=2, size=attack_count)
    
    attack_protocol = np.random.choice([1, 2, 3], attack_count, p=[0.3, 0.4, 0.3])
    attack_flag_count = np.random.poisson(lam=8, size=attack_count)  # More flags
    
    attack_df = pd.DataFrame({
        'duration': attack_duration,
        'protocol_type': attack_protocol,
        'src_bytes': attack_src_bytes,
        'dst_bytes': attack_dst_bytes,
        'flag_count': attack_flag_count,
        'packet_size_avg': attack_packet_size,
        'is_attack': True
    })
    
    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def generate_sensor_data(n_samples=15000, anomaly_rate=0.03):
    """Generate IoT sensor data with device malfunctions."""
    np.random.seed(44)
    
    # Time series with hourly readings
    start_time = datetime.now() - timedelta(days=30)
    timestamps = pd.date_range(start_time, periods=n_samples, freq='h')
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal sensor readings with daily and weekly patterns
    time_of_day = np.array([t.hour for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])
    
    # Temperature with daily cycle
    daily_temp_cycle = 10 * np.sin(2 * np.pi * time_of_day[:normal_count] / 24)
    base_temp = 20 + np.random.normal(0, 2, normal_count)
    normal_temp = base_temp + daily_temp_cycle + np.random.normal(0, 1, normal_count)
    
    # Humidity inversely correlated with temperature
    normal_humidity = 60 - 0.5 * (normal_temp - 20) + np.random.normal(0, 5, normal_count)
    normal_humidity = np.clip(normal_humidity, 10, 90)
    
    # Pressure relatively stable
    normal_pressure = np.random.normal(1013.25, 10, normal_count)
    
    # Vibration low for normal operation
    normal_vibration = np.random.exponential(scale=0.5, size=normal_count)
    
    # Power consumption with business hours pattern
    business_multiplier = np.where(
        (time_of_day[:normal_count] >= 8) & (time_of_day[:normal_count] <= 18), 1.5, 1.0
    )
    normal_power = 100 * business_multiplier + np.random.normal(0, 10, normal_count)
    
    normal_df = pd.DataFrame({
        'timestamp': timestamps[:normal_count],
        'temperature': normal_temp,
        'humidity': normal_humidity,
        'pressure': normal_pressure,
        'vibration': normal_vibration,
        'power_consumption': normal_power,
        'is_malfunction': False
    })
    
    # Malfunction data (anomalies)
    anomaly_count = n_samples - normal_count
    anomaly_timestamps = timestamps[normal_count:]
    
    # Malfunctions show extreme values
    third = anomaly_count // 3
    remainder = anomaly_count - 2 * third
    anomaly_temp = np.concatenate([
        np.random.normal(-10, 5, third),      # Too cold
        np.random.normal(50, 10, third),      # Too hot
        np.random.normal(20, 20, remainder)   # Very unstable
    ])
    
    anomaly_humidity = np.random.uniform(0, 100, anomaly_count)  # Erratic
    anomaly_pressure = np.random.normal(1013.25, 50, anomaly_count)  # Unstable
    anomaly_vibration = np.random.exponential(scale=5, size=anomaly_count)  # High
    anomaly_power = np.random.uniform(50, 300, anomaly_count)  # Erratic
    
    anomaly_df = pd.DataFrame({
        'timestamp': anomaly_timestamps,
        'temperature': anomaly_temp,
        'humidity': anomaly_humidity,
        'pressure': anomaly_pressure,
        'vibration': anomaly_vibration,
        'power_consumption': anomaly_power,
        'is_malfunction': True
    })
    
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def generate_server_metrics_data(n_samples=25000, anomaly_rate=0.04):
    """Generate server performance metrics with anomalies."""
    np.random.seed(45)
    
    # Time series (5-minute intervals)
    start_time = datetime.now() - timedelta(days=7)
    timestamps = pd.date_range(start_time, periods=n_samples, freq='5min')
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal server operation
    # CPU usage follows business hours pattern
    hour_of_day = np.array([t.hour for t in timestamps[:normal_count]])
    business_hours = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
    
    normal_cpu = 20 + 40 * business_hours + np.random.normal(0, 10, normal_count)
    normal_cpu = np.clip(normal_cpu, 0, 100)
    
    # Memory usage correlated with CPU
    normal_memory = 30 + 0.3 * normal_cpu + np.random.normal(0, 5, normal_count)
    normal_memory = np.clip(normal_memory, 10, 95)
    
    # Disk I/O related to activity
    normal_disk_io = 100 + 200 * business_hours + np.random.exponential(50, normal_count)
    normal_disk_io = np.clip(normal_disk_io, 0, 10000)
    
    # Network traffic
    normal_network = 1000 + 2000 * business_hours + np.random.exponential(500, normal_count)
    
    # Response time
    normal_response = 100 + 50 * (normal_cpu / 100) + np.random.exponential(20, normal_count)
    
    normal_df = pd.DataFrame({
        'timestamp': timestamps[:normal_count],
        'cpu_usage': normal_cpu,
        'memory_usage': normal_memory,
        'disk_io': normal_disk_io,
        'network_traffic': normal_network,
        'response_time': normal_response,
        'is_anomaly': False
    })
    
    # Anomalous server behavior
    anomaly_count = n_samples - normal_count
    
    # Various types of server anomalies
    third = anomaly_count // 3
    remainder = anomaly_count - 2 * third
    anomaly_cpu = np.concatenate([
        np.random.uniform(90, 100, third),      # CPU spikes
        np.random.uniform(0, 10, third),        # CPU drops
        np.random.uniform(20, 80, remainder)    # Normal range but other metrics off
    ])
    
    anomaly_memory = np.random.uniform(80, 98, anomaly_count)  # Memory pressure
    anomaly_disk_io = np.random.exponential(5000, anomaly_count)  # I/O storms
    anomaly_network = np.random.exponential(10000, anomaly_count)  # Traffic spikes
    anomaly_response = np.random.exponential(1000, anomaly_count)  # Slow responses
    
    anomaly_df = pd.DataFrame({
        'timestamp': timestamps[normal_count:],
        'cpu_usage': anomaly_cpu,
        'memory_usage': anomaly_memory,
        'disk_io': anomaly_disk_io,
        'network_traffic': anomaly_network,
        'response_time': anomaly_response,
        'is_anomaly': True
    })
    
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def generate_manufacturing_data(n_samples=8000, anomaly_rate=0.06):
    """Generate manufacturing quality control data."""
    np.random.seed(46)
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal manufacturing process
    # Dimensions within tolerance
    normal_length = np.random.normal(100.0, 0.5, normal_count)  # Target: 100mm ±2mm
    normal_width = np.random.normal(50.0, 0.3, normal_count)    # Target: 50mm ±1mm
    normal_height = np.random.normal(25.0, 0.2, normal_count)   # Target: 25mm ±0.5mm
    
    # Process parameters
    normal_temperature = np.random.normal(200, 5, normal_count)  # Process temp
    normal_pressure = np.random.normal(10, 1, normal_count)      # Process pressure
    normal_speed = np.random.normal(1000, 50, normal_count)      # Machine speed
    
    # Surface finish quality
    normal_surface = np.random.beta(8, 2, normal_count) * 10  # Good finish
    
    normal_df = pd.DataFrame({
        'part_id': range(normal_count),
        'length_mm': normal_length,
        'width_mm': normal_width,
        'height_mm': normal_height,
        'temperature': normal_temperature,
        'pressure': normal_pressure,
        'machine_speed': normal_speed,
        'surface_finish': normal_surface,
        'is_defective': False
    })
    
    # Defective parts
    defect_count = n_samples - normal_count
    
    # Various types of defects
    third = defect_count // 3
    remainder = defect_count - 2 * third
    defect_length = np.concatenate([
        np.random.normal(100, 3, third),        # High variation
        np.random.normal(95, 1, third),         # Systematic offset
        np.random.uniform(90, 110, remainder)   # Out of spec
    ])
    
    defect_width = np.random.normal(50, 2, defect_count)
    defect_height = np.random.normal(25, 1, defect_count)
    
    # Process parameters often off for defects
    defect_temperature = np.random.normal(200, 20, defect_count)
    defect_pressure = np.random.normal(10, 3, defect_count)
    defect_speed = np.random.normal(1000, 200, defect_count)
    
    # Poor surface finish
    defect_surface = np.random.beta(2, 8, defect_count) * 10
    
    defect_df = pd.DataFrame({
        'part_id': range(normal_count, n_samples),
        'length_mm': defect_length,
        'width_mm': defect_width,
        'height_mm': defect_height,
        'temperature': defect_temperature,
        'pressure': defect_pressure,
        'machine_speed': defect_speed,
        'surface_finish': defect_surface,
        'is_defective': True
    })
    
    df = pd.concat([normal_df, defect_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def generate_user_behavior_data(n_samples=12000, anomaly_rate=0.08):
    """Generate user behavior data with suspicious activities."""
    np.random.seed(47)
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal user behavior
    normal_session_duration = np.random.lognormal(mean=3, sigma=1, size=normal_count)
    normal_session_duration = np.clip(normal_session_duration, 1, 120)  # 1-120 minutes
    
    normal_page_views = np.random.poisson(lam=10, size=normal_count)
    normal_page_views = np.clip(normal_page_views, 1, 50)
    
    normal_clicks = normal_page_views * np.random.uniform(1, 3, normal_count)
    normal_login_frequency = np.random.exponential(scale=1, size=normal_count)  # Days between logins
    
    # Geographic and device features
    normal_countries = np.random.choice([1, 2, 3, 4, 5], normal_count, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    normal_devices = np.random.choice([1, 2, 3], normal_count, p=[0.6, 0.3, 0.1])  # Desktop, mobile, tablet
    
    # Time-based features
    hour_probs_user = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
                               0.12, 0.12, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02,
                               0.02, 0.01, 0.01, 0.01])
    hour_probs_user = hour_probs_user / hour_probs_user.sum()  # Normalize
    normal_hour = np.random.choice(range(24), normal_count, p=hour_probs_user)
    
    normal_df = pd.DataFrame({
        'user_id': range(normal_count),
        'session_duration': normal_session_duration,
        'page_views': normal_page_views,
        'clicks': normal_clicks,
        'login_frequency': normal_login_frequency,
        'country_code': normal_countries,
        'device_type': normal_devices,
        'hour_of_day': normal_hour,
        'is_suspicious': False
    })
    
    # Suspicious user behavior
    suspicious_count = n_samples - normal_count
    
    # Bot-like or malicious patterns
    third = suspicious_count // 3
    remainder = suspicious_count - 2 * third
    suspicious_session = np.concatenate([
        np.random.uniform(0.1, 2, third),           # Very short sessions
        np.random.uniform(120, 300, third),         # Very long sessions
        np.random.lognormal(3, 2, remainder)        # Highly variable
    ])
    
    suspicious_page_views = np.random.poisson(lam=50, size=suspicious_count)  # Many pages
    suspicious_clicks = suspicious_page_views * np.random.uniform(0.1, 0.5, suspicious_count)  # Few clicks per page
    suspicious_login_frequency = np.random.exponential(scale=0.1, size=suspicious_count)  # Very frequent
    
    # Unusual geographic/device patterns
    suspicious_countries = np.random.choice([1, 2, 3, 4, 5], suspicious_count)  # Uniform distribution
    suspicious_devices = np.random.choice([1, 2, 3], suspicious_count)
    suspicious_hour = np.random.choice(range(24), suspicious_count)  # Any time
    
    suspicious_df = pd.DataFrame({
        'user_id': range(normal_count, n_samples),
        'session_duration': suspicious_session,
        'page_views': suspicious_page_views,
        'clicks': suspicious_clicks,
        'login_frequency': suspicious_login_frequency,
        'country_code': suspicious_countries,
        'device_type': suspicious_devices,
        'hour_of_day': suspicious_hour,
        'is_suspicious': True
    })
    
    df = pd.concat([normal_df, suspicious_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def generate_time_series_data(n_samples=5000, anomaly_rate=0.1):
    """Generate time series data with temporal anomalies."""
    np.random.seed(48)
    
    # Time series with minute-level data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=3),
        periods=n_samples,
        freq='1min'
    )
    
    # Base signal with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.001 * t  # Slight upward trend
    daily_cycle = 10 * np.sin(2 * np.pi * t / (24 * 60))  # Daily pattern
    hourly_cycle = 3 * np.sin(2 * np.pi * t / 60)         # Hourly pattern
    noise = np.random.normal(0, 1, n_samples)
    
    base_signal = 50 + trend + daily_cycle + hourly_cycle + noise
    
    # Add anomalies
    anomaly_count = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, anomaly_count, replace=False)
    
    signal_with_anomalies = base_signal.copy()
    anomaly_labels = np.zeros(n_samples, dtype=bool)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drop', 'shift', 'noise'])
        
        if anomaly_type == 'spike':
            signal_with_anomalies[idx] += np.random.uniform(20, 50)
        elif anomaly_type == 'drop':
            signal_with_anomalies[idx] -= np.random.uniform(20, 50)
        elif anomaly_type == 'shift':
            # Level shift for several points
            shift_length = min(10, n_samples - idx)
            shift_amount = np.random.uniform(-30, 30)
            signal_with_anomalies[idx:idx+shift_length] += shift_amount
        elif anomaly_type == 'noise':
            # High noise for several points
            noise_length = min(5, n_samples - idx)
            signal_with_anomalies[idx:idx+noise_length] += np.random.normal(0, 10, noise_length)
        
        anomaly_labels[idx] = True
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': signal_with_anomalies,
        'trend_component': 50 + trend,
        'daily_component': daily_cycle,
        'hourly_component': hourly_cycle,
        'is_anomaly': anomaly_labels
    })
    
    return df


def generate_mixed_features_data(n_samples=5000, anomaly_rate=0.15):
    """Generate general-purpose dataset with mixed feature types."""
    np.random.seed(49)
    
    normal_count = int(n_samples * (1 - anomaly_rate))
    
    # Normal data with various feature types
    # Continuous features
    normal_continuous_1 = np.random.normal(0, 1, normal_count)
    normal_continuous_2 = np.random.exponential(2, normal_count)
    normal_continuous_3 = np.random.beta(2, 5, normal_count) * 100
    
    # Discrete features
    normal_discrete_1 = np.random.poisson(5, normal_count)
    normal_discrete_2 = np.random.geometric(0.3, normal_count)
    
    # Categorical features (encoded as integers)
    normal_categorical_1 = np.random.choice([1, 2, 3, 4], normal_count, p=[0.4, 0.3, 0.2, 0.1])
    normal_categorical_2 = np.random.choice([1, 2, 3], normal_count, p=[0.5, 0.3, 0.2])
    
    # Binary features
    normal_binary_1 = np.random.binomial(1, 0.3, normal_count)
    normal_binary_2 = np.random.binomial(1, 0.7, normal_count)
    
    # Correlated features
    normal_corr_base = np.random.normal(0, 1, normal_count)
    normal_corr_1 = normal_corr_base + np.random.normal(0, 0.5, normal_count)
    normal_corr_2 = -normal_corr_base + np.random.normal(0, 0.5, normal_count)
    
    normal_df = pd.DataFrame({
        'continuous_1': normal_continuous_1,
        'continuous_2': normal_continuous_2,
        'continuous_3': normal_continuous_3,
        'discrete_1': normal_discrete_1,
        'discrete_2': normal_discrete_2,
        'categorical_1': normal_categorical_1,
        'categorical_2': normal_categorical_2,
        'binary_1': normal_binary_1,
        'binary_2': normal_binary_2,
        'correlated_1': normal_corr_1,
        'correlated_2': normal_corr_2,
        'is_anomaly': False
    })
    
    # Anomalous data
    anomaly_count = n_samples - normal_count
    
    # Anomalies have different distributions
    anomaly_continuous_1 = np.random.normal(5, 2, anomaly_count)  # Different mean/std
    anomaly_continuous_2 = np.random.exponential(0.5, anomaly_count)  # Different scale
    anomaly_continuous_3 = np.random.uniform(0, 200, anomaly_count)  # Different distribution
    
    anomaly_discrete_1 = np.random.poisson(20, anomaly_count)  # Much higher values
    anomaly_discrete_2 = np.random.geometric(0.1, anomaly_count)  # Different parameter
    
    # Unusual categorical combinations
    anomaly_categorical_1 = np.random.choice([1, 2, 3, 4], anomaly_count)  # Uniform
    anomaly_categorical_2 = np.random.choice([1, 2, 3], anomaly_count)
    
    # Unusual binary patterns
    anomaly_binary_1 = np.random.binomial(1, 0.8, anomaly_count)  # Different probability
    anomaly_binary_2 = np.random.binomial(1, 0.1, anomaly_count)
    
    # Break correlations
    anomaly_corr_1 = np.random.normal(0, 3, anomaly_count)  # Independent
    anomaly_corr_2 = np.random.normal(0, 3, anomaly_count)  # Independent
    
    anomaly_df = pd.DataFrame({
        'continuous_1': anomaly_continuous_1,
        'continuous_2': anomaly_continuous_2,
        'continuous_3': anomaly_continuous_3,
        'discrete_1': anomaly_discrete_1,
        'discrete_2': anomaly_discrete_2,
        'categorical_1': anomaly_categorical_1,
        'categorical_2': anomaly_categorical_2,
        'binary_1': anomaly_binary_1,
        'binary_2': anomaly_binary_2,
        'correlated_1': anomaly_corr_1,
        'correlated_2': anomaly_corr_2,
        'is_anomaly': True
    })
    
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def main():
    """Generate all example datasets."""
    parser = argparse.ArgumentParser(description='Generate example datasets for anomaly detection tutorials')
    parser.add_argument('--output-dir', default='datasets', help='Output directory for datasets')
    parser.add_argument('--small', action='store_true', help='Generate smaller datasets for quick testing')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Scale factor for dataset sizes
    scale = 0.1 if args.small else 1.0
    
    datasets = [
        ('credit_card_transactions', generate_credit_card_data, int(10000 * scale)),
        ('network_traffic', generate_network_traffic_data, int(20000 * scale)),
        ('sensor_readings', generate_sensor_data, int(15000 * scale)),
        ('server_metrics', generate_server_metrics_data, int(25000 * scale)),
        ('manufacturing_quality', generate_manufacturing_data, int(8000 * scale)),
        ('user_behavior', generate_user_behavior_data, int(12000 * scale)),
        ('time_series_anomalies', generate_time_series_data, int(5000 * scale)),
        ('mixed_features', generate_mixed_features_data, int(5000 * scale)),
    ]
    
    print("🏭 Generating Example Datasets for Anomaly Detection")
    print("=" * 60)
    
    for name, generator_func, n_samples in datasets:
        print(f"\n📊 Generating {name}.csv...")
        
        # Generate dataset
        df = generator_func(n_samples)
        
        # Save to CSV
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        
        # Print summary
        total_samples = len(df)
        if 'is_anomaly' in df.columns:
            anomalies = df['is_anomaly'].sum()
        elif 'is_fraud' in df.columns:
            anomalies = df['is_fraud'].sum()
        elif 'is_attack' in df.columns:
            anomalies = df['is_attack'].sum()
        elif 'is_malfunction' in df.columns:
            anomalies = df['is_malfunction'].sum()
        elif 'is_defective' in df.columns:
            anomalies = df['is_defective'].sum()
        elif 'is_suspicious' in df.columns:
            anomalies = df['is_suspicious'].sum()
        else:
            anomalies = 0
        
        print(f"   ✅ Generated {total_samples:,} samples")
        if anomalies > 0:
            print(f"   🚨 Contains {anomalies:,} anomalies ({anomalies/total_samples*100:.1f}%)")
        print(f"   💾 Saved to {output_path}")
        print(f"   📋 Features: {list(df.columns)}")
    
    print(f"\n🎉 Successfully generated {len(datasets)} datasets!")
    print(f"📁 All files saved to: {output_dir.absolute()}")
    
    # Create README for datasets
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Example Datasets for Anomaly Detection

This directory contains various example datasets that can be used with the anomaly detection package tutorials and documentation.

## Available Datasets

### 1. credit_card_transactions.csv
- **Use case**: Financial fraud detection
- **Features**: transaction amount, time, merchant category, location risk
- **Anomalies**: Fraudulent transactions with unusual patterns

### 2. network_traffic.csv  
- **Use case**: Network intrusion detection
- **Features**: packet size, duration, protocol, byte counts
- **Anomalies**: Attack traffic with suspicious patterns

### 3. sensor_readings.csv
- **Use case**: IoT device monitoring
- **Features**: temperature, humidity, pressure, vibration, power
- **Anomalies**: Device malfunctions with extreme readings

### 4. server_metrics.csv
- **Use case**: IT infrastructure monitoring  
- **Features**: CPU, memory, disk I/O, network, response time
- **Anomalies**: Performance issues and system problems

### 5. manufacturing_quality.csv
- **Use case**: Quality control monitoring
- **Features**: part dimensions, process parameters, surface finish
- **Anomalies**: Defective parts outside specifications

### 6. user_behavior.csv
- **Use case**: User activity analysis
- **Features**: session duration, page views, clicks, login patterns
- **Anomalies**: Suspicious user behavior (bots, malicious activity)

### 7. time_series_anomalies.csv
- **Use case**: Temporal anomaly detection
- **Features**: time-indexed values with trend and seasonality
- **Anomalies**: Spikes, drops, level shifts, noise bursts

### 8. mixed_features.csv
- **Use case**: General purpose anomaly detection
- **Features**: Mix of continuous, discrete, categorical, binary features
- **Anomalies**: Points with unusual feature combinations

## Usage

These datasets can be used directly with the quickstart templates:

```python
import pandas as pd
from anomaly_detection import DetectionService

# Load any dataset
data = pd.read_csv('datasets/credit_card_transactions.csv')

# Use numeric features for detection
numeric_features = data.select_dtypes(include=['number']).values

# Detect anomalies
service = DetectionService()
result = service.detect(numeric_features, algorithm='isolation_forest')
```

## Generation

To regenerate these datasets with different parameters:

```bash
python generate_example_data.py
python generate_example_data.py --small  # For smaller test datasets
```
""")
    
    print(f"📖 Created dataset documentation: {readme_path}")


if __name__ == "__main__":
    main()