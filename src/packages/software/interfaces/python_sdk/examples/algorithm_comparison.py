#!/usr/bin/env python3
"""
Algorithm Comparison Example

This example demonstrates how to compare multiple anomaly detection algorithms
on the same dataset to find the best performer.
"""

import asyncio

# Add parent directory to path for imports
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from pynomaly_detection.domain.entities import Dataset, Detector
from pynomaly_detection.infrastructure.config import create_container


def create_sample_data(
    n_samples: int = 500, n_features: int = 10, anomaly_fraction: float = 0.1
):
    """Create a dataset with controlled anomalies."""
    np.random.seed(42)

    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies

    # Normal data: multivariate normal distribution
    normal_data = np.random.randn(n_normal, n_features)

    # Anomalies: samples from different distributions
    anomaly_types = [
        # Shifted mean
        lambda: np.random.randn(n_features) + 3,
        # Different covariance
        lambda: np.random.randn(n_features) * 3,
        # Uniform distribution
        lambda: np.random.uniform(-4, 4, n_features),
    ]

    anomalies = []
    for i in range(n_anomalies):
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        anomalies.append(anomaly_type())

    anomalies = np.array(anomalies)

    # Combine and shuffle
    data = np.vstack([normal_data, anomalies])
    labels = np.array([0] * n_normal + [1] * n_anomalies)

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Create DataFrame
    columns = [f"feature_{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    return df, labels


async def evaluate_algorithm(
    container,
    dataset: Dataset,
    algorithm: str,
    parameters: dict[str, Any],
    true_labels: np.ndarray,
):
    """Evaluate a single algorithm and return metrics."""
    # Create detector
    detector = Detector(
        name=f"{algorithm} Detector", algorithm=algorithm, parameters=parameters
    )

    # Save detector
    detector_repo = container.detector_repository()
    detector_repo.save(detector)

    # Get detection service
    detection_service = container.detection_service()

    # Train detector
    start_time = time.time()
    await detection_service.train_detector(detector_id=detector.id, dataset=dataset)
    training_time = time.time() - start_time

    # Detect anomalies
    start_time = time.time()
    result = await detection_service.detect_anomalies(
        detector_id=detector.id, dataset=dataset
    )
    detection_time = time.time() - start_time

    # Calculate metrics if we have true labels
    predictions = np.zeros(len(dataset.data))
    predictions[result.anomaly_indices] = 1

    # True Positives, False Positives, etc.
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / len(predictions)

    return {
        "algorithm": algorithm,
        "anomalies_found": result.n_anomalies,
        "anomaly_rate": result.anomaly_rate,
        "training_time": training_time,
        "detection_time": detection_time,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


async def main():
    """Compare multiple anomaly detection algorithms."""
    print("🔍 Pynomaly Algorithm Comparison Example\n")

    # Initialize container
    container = create_container()

    # Create dataset
    print("1. Creating dataset...")
    data, true_labels = create_sample_data(
        n_samples=500, n_features=10, anomaly_fraction=0.1
    )
    print(f"   Dataset: {len(data)} samples, {data.shape[1]} features")
    print(
        f"   True anomalies: {np.sum(true_labels)} ({np.mean(true_labels) * 100:.1f}%)"
    )

    # Create dataset entity
    dataset = Dataset(
        name="Comparison Dataset",
        data=data,
        metadata={
            "n_true_anomalies": int(np.sum(true_labels)),
            "anomaly_fraction": float(np.mean(true_labels)),
        },
    )

    # Save dataset
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    # Define algorithms to compare
    algorithms = [
        {
            "name": "IsolationForest",
            "params": {"contamination": 0.1, "n_estimators": 100, "random_state": 42},
        },
        {"name": "LOF", "params": {"contamination": 0.1, "n_neighbors": 20}},
        {"name": "OCSVM", "params": {"nu": 0.1, "kernel": "rbf", "gamma": "auto"}},
        {"name": "COPOD", "params": {"contamination": 0.1}},
        {"name": "ECOD", "params": {"contamination": 0.1}},
        {"name": "KNN", "params": {"contamination": 0.1, "n_neighbors": 5}},
    ]

    # Evaluate each algorithm
    print("\n2. Evaluating algorithms...")
    results = []

    for algo in algorithms:
        print(f"   Testing {algo['name']}...")
        try:
            metrics = await evaluate_algorithm(
                container, dataset, algo["name"], algo["params"], true_labels
            )
            results.append(metrics)
        except Exception as e:
            print(f"   ⚠️  Error with {algo['name']}: {str(e)}")

    # Display results
    print("\n3. Results Summary:")
    print("-" * 100)
    print(
        f"{'Algorithm':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Time (s)':<10}"
    )
    print("-" * 100)

    # Sort by F1 score
    results.sort(key=lambda x: x["f1_score"], reverse=True)

    for r in results:
        total_time = r["training_time"] + r["detection_time"]
        print(
            f"{r['algorithm']:<20} {r['precision']:<10.3f} {r['recall']:<10.3f} "
            f"{r['f1_score']:<10.3f} {r['accuracy']:<10.3f} {total_time:<10.3f}"
        )

    # Best performer
    if results:
        best = results[0]
        print(
            f"\n✅ Best performer: {best['algorithm']} (F1-Score: {best['f1_score']:.3f})"
        )

        # Detailed metrics for best
        print(f"\n   Detailed metrics for {best['algorithm']}:")
        print(f"   - True Positives:  {best['true_positives']}")
        print(f"   - False Positives: {best['false_positives']}")
        print(f"   - True Negatives:  {best['true_negatives']}")
        print(f"   - False Negatives: {best['false_negatives']}")

    print("\n📊 Algorithm comparison completed!")


if __name__ == "__main__":
    asyncio.run(main())
