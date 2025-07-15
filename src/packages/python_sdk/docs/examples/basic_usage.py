"""
Basic Usage Examples for Pynomaly Python SDK

This file demonstrates how to use the Pynomaly Python SDK for
anomaly detection tasks.
"""

import asyncio
import numpy as np
from typing import List

# Import SDK components
from python_sdk.domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType
from python_sdk.domain.value_objects.detection_metadata import DetectionMetadata
from python_sdk.domain.entities.detection_request import DetectionRequest
from python_sdk.infrastructure.adapters.pyod_algorithm_adapter import PyODAlgorithmAdapter


async def basic_anomaly_detection():
    """
    Example 1: Basic anomaly detection with Isolation Forest
    """
    print("Example 1: Basic Anomaly Detection")
    print("=" * 40)
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 95).tolist()
    outliers = [10, -8, 12].tolist()
    data = normal_data + outliers
    
    print(f"Data size: {len(data)}")
    print(f"Expected outliers: {len(outliers)}")
    
    # Configure Isolation Forest algorithm
    algorithm_config = AlgorithmConfig(
        algorithm_type=AlgorithmType.ISOLATION_FOREST,
        parameters={"n_estimators": 100, "max_samples": "auto"},
        contamination=0.05,  # Expect 5% outliers
        random_state=42
    )
    
    # Create detection request
    metadata = DetectionMetadata(
        dataset_name="sample_data",
        environment="development"
    )
    
    request = DetectionRequest(
        data=data,
        algorithm_config=algorithm_config,
        metadata=metadata
    )
    
    print(f"Request ID: {request.id}")
    print(f"Algorithm: {algorithm_config.algorithm_type.value}")
    
    # Execute detection
    adapter = PyODAlgorithmAdapter()
    result = await adapter.detect_anomalies(data, algorithm_config)
    
    # Display results
    anomaly_indices = [i for i, is_anomaly in enumerate(result.anomalies) if is_anomaly]
    print(f"Anomalies found at indices: {anomaly_indices}")
    print(f"Execution time: {result.execution_time_ms}ms")
    print()


async def compare_algorithms():
    """
    Example 2: Compare different algorithms on the same data
    """
    print("Example 2: Algorithm Comparison")
    print("=" * 40)
    
    # Create data with clear outliers
    np.random.seed(42)
    data = np.random.normal(0, 1, 100).tolist()
    data.extend([5, -5, 6, -6])  # Add clear outliers
    
    algorithms = [
        (AlgorithmType.ISOLATION_FOREST, {"n_estimators": 100}),
        (AlgorithmType.LOCAL_OUTLIER_FACTOR, {"n_neighbors": 20}),
        (AlgorithmType.ELLIPTIC_ENVELOPE, {})
    ]
    
    adapter = PyODAlgorithmAdapter()
    
    print(f"Testing {len(algorithms)} algorithms on {len(data)} data points")
    print()
    
    for algo_type, params in algorithms:
        config = AlgorithmConfig(
            algorithm_type=algo_type,
            parameters=params,
            contamination=0.05
        )
        
        try:
            result = await adapter.detect_anomalies(data, config)
            anomaly_count = sum(result.anomalies)
            
            print(f"Algorithm: {algo_type.value}")
            print(f"  Anomalies detected: {anomaly_count}")
            print(f"  Execution time: {result.execution_time_ms}ms")
            print(f"  Average score: {np.mean(result.scores):.3f}")
            print()
            
        except Exception as e:
            print(f"Algorithm {algo_type.value} failed: {str(e)}")
            print()


async def custom_algorithm_parameters():
    """
    Example 3: Using custom algorithm parameters
    """
    print("Example 3: Custom Algorithm Parameters")
    print("=" * 40)
    
    # Generate time series data with anomalies
    np.random.seed(42)
    time_series = []
    
    # Normal pattern
    for i in range(100):
        value = np.sin(i * 0.1) + np.random.normal(0, 0.1)
        time_series.append(value)
    
    # Add anomalies
    time_series[25] = 5  # Spike
    time_series[50] = -4  # Dip
    time_series[75] = 3   # Another spike
    
    print(f"Time series length: {len(time_series)}")
    
    # Configure LOF with custom parameters
    lof_config = AlgorithmConfig(
        algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
        parameters={
            "n_neighbors": 10,  # Smaller neighborhood
            "algorithm": "auto",
            "metric": "euclidean"
        },
        contamination=0.03  # Expect 3% outliers
    )
    
    adapter = PyODAlgorithmAdapter()
    result = await adapter.detect_anomalies(time_series, lof_config)
    
    # Find anomaly positions
    anomaly_positions = [i for i, is_anomaly in enumerate(result.anomalies) if is_anomaly]
    
    print(f"LOF Parameters: {lof_config.parameters}")
    print(f"Anomalies detected at positions: {anomaly_positions}")
    print(f"Expected anomalies at: [25, 50, 75]")
    print()


async def algorithm_validation():
    """
    Example 4: Algorithm configuration validation
    """
    print("Example 4: Algorithm Validation")
    print("=" * 40)
    
    # Test valid configuration
    try:
        valid_config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 200, "max_samples": 0.8},
            contamination=0.1,
            random_state=42
        )
        print("✓ Valid configuration created successfully")
        print(f"  Algorithm: {valid_config.algorithm_type.value}")
        print(f"  Valid: {valid_config.is_valid()}")
        
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    print()
    
    # Test invalid configuration
    try:
        invalid_config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"invalid_param": 123},  # Invalid parameter
            contamination=0.1
        )
        print("✗ Invalid configuration should have failed!")
        
    except ValueError as e:
        print(f"✓ Validation correctly caught error: {e}")
    
    print()


async def performance_estimation():
    """
    Example 5: Performance estimation for different data sizes
    """
    print("Example 5: Performance Estimation")
    print("=" * 40)
    
    adapter = PyODAlgorithmAdapter()
    data_sizes = [100, 1000, 5000, 10000]
    
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.ISOLATION_FOREST,
        parameters={"n_estimators": 100},
        contamination=0.1
    )
    
    print("Estimated execution times:")
    for size in data_sizes:
        estimated_time = await adapter.estimate_execution_time(size, config)
        print(f"  {size:,} points: ~{estimated_time}ms")
    
    print()


async def main():
    """Run all examples."""
    print("Pynomaly Python SDK - Usage Examples")
    print("=" * 50)
    print()
    
    try:
        await basic_anomaly_detection()
        await compare_algorithms()
        await custom_algorithm_parameters()
        await algorithm_validation()
        await performance_estimation()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Note: This requires PyOD to be installed
    # pip install pyod scikit-learn numpy
    
    asyncio.run(main())