"""Configuration for benchmark tests."""

import pytest
import pandas as pd
import numpy as np
from typing import Generator, List

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    """Create a small dataset for quick benchmarks."""
    np.random.seed(42)
    
    # Generate 1000 samples
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 800)
    anomaly_data = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 200)
    
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(800), np.ones(200)])
    
    df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
    df['label'] = labels
    
    return Dataset(
        name="small_benchmark_dataset",
        data=df,
        target_column="label"
    )


@pytest.fixture(scope="session") 
def medium_dataset() -> Dataset:
    """Create a medium dataset for benchmarks."""
    np.random.seed(42)
    
    # Generate 10,000 samples with 10 features
    n_samples = 10000
    n_features = 10
    n_anomalies = 1000
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples - n_anomalies
    )
    
    # Anomalous data 
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 0.5,
        size=n_anomalies
    )
    
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_samples - n_anomalies), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = labels
    
    return Dataset(
        name="medium_benchmark_dataset",
        data=df,
        target_column="label"
    )


@pytest.fixture(scope="session")
def large_dataset() -> Dataset:
    """Create a large dataset for performance benchmarks."""
    np.random.seed(42)
    
    # Generate 50,000 samples with 20 features
    n_samples = 50000
    n_features = 20
    n_anomalies = 5000
    
    # Normal data with some correlation structure
    cov_matrix = np.eye(n_features)
    for i in range(n_features - 1):
        cov_matrix[i, i + 1] = 0.3
        cov_matrix[i + 1, i] = 0.3
    
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov_matrix,
        size=n_samples - n_anomalies
    )
    
    # Anomalous data with different distribution
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 2,
        cov=np.eye(n_features) * 0.3,
        size=n_anomalies
    )
    
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_samples - n_anomalies), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = labels
    
    return Dataset(
        name="large_benchmark_dataset",
        data=df,
        target_column="label"
    )


@pytest.fixture(scope="session")
def benchmark_algorithms() -> List[dict]:
    """Define algorithms for benchmarking."""
    return [
        {
            "name": "IsolationForest_PyOD",
            "adapter_class": PyODAdapter,
            "algorithm_name": "IsolationForest",
            "params": {"n_estimators": 100, "random_state": 42}
        },
        {
            "name": "LOF_PyOD", 
            "adapter_class": PyODAdapter,
            "algorithm_name": "LOF",
            "params": {"n_neighbors": 20}
        },
        {
            "name": "OCSVM_PyOD",
            "adapter_class": PyODAdapter,
            "algorithm_name": "OCSVM",
            "params": {"gamma": "scale"}
        },
        {
            "name": "IsolationForest_Sklearn",
            "adapter_class": SklearnAdapter,
            "algorithm_name": "IsolationForest", 
            "params": {"n_estimators": 100, "random_state": 42}
        },
        {
            "name": "LocalOutlierFactor_Sklearn",
            "adapter_class": SklearnAdapter,
            "algorithm_name": "LocalOutlierFactor",
            "params": {"n_neighbors": 20}
        }
    ]


@pytest.fixture
def performance_thresholds() -> dict:
    """Define performance thresholds for benchmarks."""
    return {
        "small_dataset": {
            "max_fit_time": 5.0,  # seconds
            "max_predict_time": 1.0,  # seconds
            "max_memory_mb": 100,  # MB
        },
        "medium_dataset": {
            "max_fit_time": 30.0,
            "max_predict_time": 5.0,
            "max_memory_mb": 500,
        },
        "large_dataset": {
            "max_fit_time": 120.0,
            "max_predict_time": 15.0,
            "max_memory_mb": 2000,
        }
    }