#!/usr/bin/env python3
"""
Performance testing script for Pynomaly anomaly detection pipeline.
Tests the system with synthetic datasets of varying sizes and characteristics.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import asyncio
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detector import Detector
from monorepo.domain.value_objects.contamination_rate import ContaminationRate
from monorepo.application.use_cases.detect_anomalies import DetectAnomaliesRequest
from monorepo.infrastructure.config.container import Container


def generate_synthetic_data(n_samples: int, n_features: int, contamination: float):
    """Generate synthetic data with outliers for testing."""
    # Generate normal data
    normal_samples = int(n_samples * (1 - contamination))
    outlier_samples = n_samples - normal_samples
    
    # Normal data - clustered
    normal_data, _ = make_blobs(
        n_samples=normal_samples,
        centers=3,
        n_features=n_features,
        cluster_std=1.0,
        random_state=42
    )
    
    # Outlier data - scattered
    outlier_data = np.random.uniform(
        low=normal_data.min() - 3,
        high=normal_data.max() + 3,
        size=(outlier_samples, n_features)
    )
    
    # Combine data
    data = np.vstack([normal_data, outlier_data])
    labels = np.hstack([
        np.zeros(normal_samples),  # Normal points
        np.ones(outlier_samples)   # Outliers
    ])
    
    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    # Normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data, labels


async def benchmark_detection_pipeline(
    n_samples: int,
    n_features: int,
    contamination: float = 0.1,
    algorithm: str = "isolation_forest"
):
    """Benchmark the complete anomaly detection pipeline."""
    print(f"\n📊 Testing with {n_samples:,} samples, {n_features} features, {contamination:.1%} contamination")
    
    # Generate test data
    start_time = time.time()
    data, true_labels = generate_synthetic_data(n_samples, n_features, contamination)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    data_gen_time = time.time() - start_time
    
    print(f"   ⏱️  Data generation: {data_gen_time:.3f}s")
    
    # Create domain entities
    start_time = time.time()
    dataset = Dataset(name=f"Synthetic_{n_samples}_{n_features}", data=df)
    # Map algorithm names to supported ones
    algorithm_mapping = {
        "isolation_forest": "IsolationForest"
    }
    supported_algorithm = algorithm_mapping.get(algorithm, algorithm)
    
    detector = Detector(
        name=f"Test_{algorithm}",
        algorithm_name=supported_algorithm,
        contamination_rate=ContaminationRate(value=contamination)
    )
    entity_time = time.time() - start_time
    
    print(f"   ⏱️  Entity creation: {entity_time:.3f}s")
    
    # Initialize container and train detector
    start_time = time.time()
    container = Container()
    detector_repo = container.detector_repository()
    await detector_repo.save(detector)
    
    # Train the detector first
    from monorepo.application.use_cases.train_detector import TrainDetectorRequest
    train_use_case = container.train_detector_use_case()
    train_request = TrainDetectorRequest(
        training_data=dataset,
        detector_id=detector.id
    )
    await train_use_case.execute(train_request)
    
    detect_use_case = container.detect_anomalies_use_case()
    setup_time = time.time() - start_time
    
    print(f"   ⏱️  Setup time: {setup_time:.3f}s")
    
    # Execute detection
    start_time = time.time()
    request = DetectAnomaliesRequest(
        dataset=dataset,
        detector_id=detector.id
    )
    
    try:
        result = await detect_use_case.execute(request)
        detection_time = time.time() - start_time
        
        print(f"   ⏱️  Detection time: {detection_time:.3f}s")
        print(f"   📈 Throughput: {n_samples / detection_time:.0f} samples/sec")
        print(f"   🎯 Detected {result.detection_result.n_anomalies} anomalies (rate: {result.detection_result.anomaly_rate:.1%})")
        
        # Calculate basic accuracy metrics if we have true labels
        detection_labels = result.detection_result.labels
        if len(true_labels) == len(detection_labels):
            tp = np.sum((true_labels == 1) & (detection_labels == 1))
            fp = np.sum((true_labels == 0) & (detection_labels == 1))
            fn = np.sum((true_labels == 1) & (detection_labels == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"   🎯 Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "contamination": contamination,
            "algorithm": algorithm,
            "data_gen_time": data_gen_time,
            "entity_time": entity_time,
            "setup_time": setup_time,
            "detection_time": detection_time,
            "throughput": n_samples / detection_time,
            "n_anomalies": result.detection_result.n_anomalies,
            "anomaly_rate": result.detection_result.anomaly_rate
        }
        
    except Exception as e:
        print(f"   ❌ Error during detection: {e}")
        return None


async def run_performance_tests():
    """Run comprehensive performance tests."""
    print("🚀 Starting Pynomaly Performance Tests")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        # Small datasets
        {"n_samples": 1000, "n_features": 5, "contamination": 0.1},
        {"n_samples": 1000, "n_features": 10, "contamination": 0.1},
        
        # Medium datasets
        {"n_samples": 10000, "n_features": 10, "contamination": 0.1},
        {"n_samples": 10000, "n_features": 20, "contamination": 0.1},
        
        # Large datasets
        {"n_samples": 50000, "n_features": 10, "contamination": 0.1},
        {"n_samples": 100000, "n_features": 5, "contamination": 0.1},
        
        # High-dimensional
        {"n_samples": 5000, "n_features": 50, "contamination": 0.1},
        
        # Different contamination rates
        {"n_samples": 10000, "n_features": 10, "contamination": 0.05},
        {"n_samples": 10000, "n_features": 10, "contamination": 0.2},
    ]
    
    results = []
    algorithms = ["isolation_forest"]  # Can expand to test multiple algorithms
    
    total_start_time = time.time()
    
    for algorithm in algorithms:
        print(f"\n🔧 Testing algorithm: {algorithm}")
        print("-" * 30)
        
        for config in test_configs:
            result = await benchmark_detection_pipeline(
                algorithm=algorithm,
                **config
            )
            if result:
                results.append(result)
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n🏁 Performance Test Summary")
    print("=" * 50)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Tests completed: {len(results)}")
    
    if results:
        avg_throughput = np.mean([r["throughput"] for r in results])
        max_throughput = max(r["throughput"] for r in results)
        min_throughput = min(r["throughput"] for r in results)
        
        print(f"Average throughput: {avg_throughput:.0f} samples/sec")
        print(f"Max throughput: {max_throughput:.0f} samples/sec")
        print(f"Min throughput: {min_throughput:.0f} samples/sec")
        
        # Find best performing configuration
        best_result = max(results, key=lambda r: r["throughput"])
        print(f"\n🏆 Best performance:")
        print(f"   {best_result['n_samples']:,} samples, {best_result['n_features']} features")
        print(f"   {best_result['throughput']:.0f} samples/sec")
        
        # Find worst performing configuration
        worst_result = min(results, key=lambda r: r["throughput"])
        print(f"\n⚠️  Slowest performance:")
        print(f"   {worst_result['n_samples']:,} samples, {worst_result['n_features']} features")
        print(f"   {worst_result['throughput']:.0f} samples/sec")
        
    print(f"\n✅ Performance testing completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_performance_tests())