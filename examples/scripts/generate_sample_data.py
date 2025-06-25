#!/usr/bin/env python
"""Generate sample datasets for testing and demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_simple_anomalies(
    n_samples: int = 1000,
    n_features: int = 2,
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate simple dataset with anomalies."""
    np.random.seed(random_state)
    
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Generate normal data (centered around 0)
    normal_data = np.random.randn(n_normal, n_features)
    
    # Generate anomalies (far from center)
    anomaly_data = np.random.randn(n_anomalies, n_features) * 3 + 5
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.array([0] * n_normal + [1] * n_anomalies)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y
    
    return df


def generate_time_series_anomalies(
    n_samples: int = 1000,
    contamination: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate time series data with anomalies."""
    np.random.seed(random_state)
    
    # Time index
    time = pd.date_range(start="2024-01-01", periods=n_samples, freq="H")
    
    # Base signal (sine wave with trend)
    trend = np.linspace(100, 150, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily pattern
    noise = np.random.randn(n_samples) * 2
    
    signal = trend + seasonal + noise
    
    # Add anomalies
    n_anomalies = int(n_samples * contamination)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Different types of anomalies
    for i, idx in enumerate(anomaly_indices):
        if i % 3 == 0:
            # Point anomaly (spike)
            signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(20, 40)
        elif i % 3 == 1:
            # Level shift
            signal[idx:min(idx+10, n_samples)] += np.random.uniform(15, 25)
        else:
            # Variance change
            signal[max(0, idx-5):min(idx+5, n_samples)] *= np.random.uniform(1.5, 2.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": time,
        "value": signal,
        "hour": time.hour,
        "day_of_week": time.dayofweek,
        "is_weekend": (time.dayofweek >= 5).astype(int)
    })
    
    # Add label
    df["label"] = 0
    df.loc[anomaly_indices, "label"] = 1
    
    return df


def generate_clustering_anomalies(
    n_samples: int = 1000,
    n_clusters: int = 3,
    n_features: int = 2,
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate clustered data with anomalies."""
    np.random.seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    n_anomalies = int(n_samples * contamination)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features) * 10
    
    # Generate clustered data
    X = []
    cluster_labels = []
    
    for i in range(n_clusters):
        cluster_data = np.random.randn(samples_per_cluster, n_features) * 0.5 + centers[i]
        X.append(cluster_data)
        cluster_labels.extend([i] * samples_per_cluster)
    
    X = np.vstack(X)
    
    # Generate anomalies (between clusters)
    anomaly_data = np.random.uniform(
        X.min(axis=0) - 5,
        X.max(axis=0) + 5,
        size=(n_anomalies, n_features)
    )
    
    # Remove some normal points and add anomalies
    X = X[:-n_anomalies]
    cluster_labels = cluster_labels[:-n_anomalies]
    
    X = np.vstack([X, anomaly_data])
    cluster_labels.extend([-1] * n_anomalies)  # -1 for anomalies
    
    # Create labels
    y = np.array([0] * (len(X) - n_anomalies) + [1] * n_anomalies)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    cluster_labels = [cluster_labels[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["cluster"] = cluster_labels
    df["label"] = y
    
    return df


def generate_high_dimensional_anomalies(
    n_samples: int = 1000,
    n_features: int = 50,
    n_informative: int = 10,
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate high-dimensional data with anomalies."""
    np.random.seed(random_state)
    
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Generate informative features
    informative_normal = np.random.randn(n_normal, n_informative)
    informative_anomaly = np.random.randn(n_anomalies, n_informative) * 2 + 3
    
    # Generate noise features
    noise_normal = np.random.randn(n_normal, n_features - n_informative) * 0.1
    noise_anomaly = np.random.randn(n_anomalies, n_features - n_informative) * 0.1
    
    # Combine
    normal_data = np.hstack([informative_normal, noise_normal])
    anomaly_data = np.hstack([informative_anomaly, noise_anomaly])
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.array([0] * n_normal + [1] * n_anomalies)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y
    
    return df


def main():
    """Generate sample datasets."""
    parser = argparse.ArgumentParser(description="Generate sample datasets for Pynomaly")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_data"),
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["simple", "timeseries", "clustering", "highdim", "all"],
        default=["all"],
        help="Types of datasets to generate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "simple": generate_simple_anomalies,
        "timeseries": generate_time_series_anomalies,
        "clustering": generate_clustering_anomalies,
        "highdim": generate_high_dimensional_anomalies
    }
    
    if "all" in args.types:
        types_to_generate = list(datasets.keys())
    else:
        types_to_generate = args.types
    
    for dataset_type in types_to_generate:
        print(f"Generating {dataset_type} dataset...")
        
        # Generate dataset
        df = datasets[dataset_type]()
        
        # Save to CSV
        output_path = args.output_dir / f"{dataset_type}_anomalies.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  Saved to: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Anomalies: {df['label'].sum()} ({df['label'].mean():.1%})")
        print()
    
    print("Done! Generated datasets in:", args.output_dir)


if __name__ == "__main__":
    main()