#!/usr/bin/env python3
"""
Ensemble Detection Example

This example shows how to combine multiple anomaly detectors into an ensemble
for more robust anomaly detection. It demonstrates different voting strategies.
"""

import asyncio

# Add parent directory to path for imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config import create_container


def create_complex_dataset():
    """Create a dataset with different types of anomalies."""
    np.random.seed(42)

    # Base normal data
    n_normal = 300
    normal_data = np.random.randn(n_normal, 5)

    # Different types of anomalies
    anomaly_types = {
        "global_outliers": np.random.randn(10, 5) * 5,  # Far from center
        "local_outliers": np.random.randn(10, 5)
        + [2, 0, 0, 0, 0],  # Shifted in one dimension
        "cluster_anomalies": np.random.randn(10, 5) * 0.1
        + [3, 3, 0, 0, 0],  # Small cluster away
        "dependency_anomalies": np.array(
            [  # Break correlation patterns
                [x, -x, x, -x, x] + np.random.randn(5) * 0.1
                for x in np.linspace(2, 4, 10)
            ]
        ),
    }

    # Combine all data
    all_anomalies = np.vstack(list(anomaly_types.values()))
    data = np.vstack([normal_data, all_anomalies])

    # Create labels for evaluation
    labels = np.zeros(len(data))
    labels[n_normal:] = 1  # Mark anomalies

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Create DataFrame
    columns = [f"feature_{i + 1}" for i in range(5)]
    df = pd.DataFrame(data, columns=columns)

    return df, labels, anomaly_types


async def create_ensemble_detectors(container, dataset: Dataset) -> list[str]:
    """Create and train multiple detectors for ensemble."""
    detector_configs = [
        # Good for global outliers
        {
            "name": "IsolationForest (Global)",
            "algorithm": "IsolationForest",
            "params": {"contamination": 0.1, "n_estimators": 100},
        },
        # Good for local outliers
        {
            "name": "LOF (Local)",
            "algorithm": "LOF",
            "params": {"contamination": 0.1, "n_neighbors": 20},
        },
        # Good for density-based anomalies
        {
            "name": "KNN (Density)",
            "algorithm": "KNN",
            "params": {"contamination": 0.1, "n_neighbors": 10},
        },
        # Good for statistical outliers
        {
            "name": "COPOD (Statistical)",
            "algorithm": "COPOD",
            "params": {"contamination": 0.1},
        },
        # Good for linear dependencies
        {
            "name": "PCA (Linear)",
            "algorithm": "PCA",
            "params": {"contamination": 0.1, "n_components": 3},
        },
    ]

    detector_ids = []
    detector_repo = container.detector_repository()
    detection_service = container.detection_service()

    print("\n2. Creating and training ensemble detectors...")

    for config in detector_configs:
        print(f"   Training {config['name']}...")

        # Create detector
        detector = Detector(
            name=config["name"],
            algorithm=config["algorithm"],
            parameters=config["params"],
        )
        detector_repo.save(detector)

        # Train detector
        await detection_service.train_detector(detector_id=detector.id, dataset=dataset)

        detector_ids.append(detector.id)

    return detector_ids


async def ensemble_detect(
    container, detector_ids: list[str], dataset: Dataset, method: str = "voting"
):
    """Perform ensemble detection using specified method."""
    detection_service = container.detection_service()
    container.ensemble_service()

    # Get individual predictions
    all_predictions = []
    all_scores = []

    for detector_id in detector_ids:
        result = await detection_service.detect_anomalies(
            detector_id=detector_id, dataset=dataset
        )

        # Create binary predictions
        predictions = np.zeros(len(dataset.data))
        predictions[result.anomaly_indices] = 1
        all_predictions.append(predictions)

        # Store anomaly scores if available
        if hasattr(result, "anomaly_scores") and result.anomaly_scores is not None:
            all_scores.append(result.anomaly_scores)

    all_predictions = np.array(all_predictions)

    # Apply ensemble method
    if method == "voting":
        # Majority voting
        ensemble_predictions = (
            np.sum(all_predictions, axis=0) >= len(detector_ids) / 2
        ).astype(int)
    elif method == "unanimous":
        # All detectors must agree
        ensemble_predictions = (
            np.sum(all_predictions, axis=0) == len(detector_ids)
        ).astype(int)
    elif method == "any":
        # Any detector flags as anomaly
        ensemble_predictions = (np.sum(all_predictions, axis=0) > 0).astype(int)
    elif method == "average_score" and all_scores:
        # Average anomaly scores
        avg_scores = np.mean(all_scores, axis=0)
        threshold = np.percentile(avg_scores, 90)  # Top 10%
        ensemble_predictions = (avg_scores > threshold).astype(int)
    else:
        # Default to voting
        ensemble_predictions = (
            np.sum(all_predictions, axis=0) >= len(detector_ids) / 2
        ).astype(int)

    # Get anomaly indices
    anomaly_indices = np.where(ensemble_predictions == 1)[0]

    return anomaly_indices, all_predictions


def evaluate_predictions(true_labels, predictions):
    """Calculate evaluation metrics."""
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


async def main():
    """Demonstrate ensemble anomaly detection."""
    print("🔍 Pynomaly Ensemble Detection Example\n")

    # Initialize container
    container = create_container()

    # Create dataset
    print("1. Creating complex dataset with multiple anomaly types...")
    data, true_labels, anomaly_types = create_complex_dataset()
    print(f"   Dataset: {len(data)} samples, {data.shape[1]} features")
    print(
        f"   True anomalies: {np.sum(true_labels)} ({np.mean(true_labels) * 100:.1f}%)"
    )
    print("   Anomaly types included:")
    for atype, adata in anomaly_types.items():
        print(f"   - {atype}: {len(adata)} samples")

    # Create dataset entity
    dataset = Dataset(
        name="Ensemble Dataset",
        data=data,
        metadata={
            "anomaly_types": list(anomaly_types.keys()),
            "n_true_anomalies": int(np.sum(true_labels)),
        },
    )

    # Save dataset
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    # Create ensemble
    detector_ids = await create_ensemble_detectors(container, dataset)

    # Test different ensemble methods
    print("\n3. Testing ensemble methods...")
    ensemble_methods = ["voting", "unanimous", "any"]

    results = {}
    for method in ensemble_methods:
        print(f"\n   Method: {method}")

        # Get ensemble predictions
        anomaly_indices, individual_predictions = await ensemble_detect(
            container, detector_ids, dataset, method
        )

        # Create prediction array
        predictions = np.zeros(len(data))
        predictions[anomaly_indices] = 1

        # Evaluate
        metrics = evaluate_predictions(true_labels, predictions)
        results[method] = metrics

        print(f"   - Anomalies detected: {len(anomaly_indices)}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - F1-Score: {metrics['f1_score']:.3f}")

    # Compare individual vs ensemble
    print("\n4. Individual detector performance:")
    for i, detector_id in enumerate(detector_ids):
        detector = container.detector_repository().get(detector_id)
        individual_metrics = evaluate_predictions(
            true_labels, individual_predictions[i]
        )
        print(f"   {detector.name}: F1={individual_metrics['f1_score']:.3f}")

    # Best ensemble method
    best_method = max(results.items(), key=lambda x: x[1]["f1_score"])
    print(
        f"\n✅ Best ensemble method: {best_method[0]} (F1-Score: {best_method[1]['f1_score']:.3f})"
    )

    print("\n📊 Ensemble detection completed!")


if __name__ == "__main__":
    asyncio.run(main())
