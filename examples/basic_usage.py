#!/usr/bin/env python3
"""
Basic Pynomaly Usage Example

This example demonstrates the simplest way to use Pynomaly for anomaly detection.
It shows how to:
1. Load data
2. Create a detector
3. Train the model
4. Detect anomalies
5. View results
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


def create_sample_data():
    """Create a simple 2D dataset with some anomalies."""
    np.random.seed(42)

    # Normal data points
    normal_data = np.random.randn(100, 2)

    # Add some anomalies
    anomalies = np.array(
        [
            [5, 5],  # Far from center
            [-5, -5],  # Far from center
            [0, 6],  # Outlier in one dimension
            [6, 0],  # Outlier in other dimension
        ]
    )

    # Combine data
    data = np.vstack([normal_data, anomalies])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["feature1", "feature2"])

    return df


async def main():
    """Main function demonstrating basic anomaly detection workflow."""
    print("🔍 Pynomaly Basic Usage Example\n")

    # Initialize container
    container = create_container()

    # Step 1: Create sample data
    print("1. Creating sample dataset...")
    data = create_sample_data()
    print(f"   Created dataset with {len(data)} samples and {data.shape[1]} features")

    # Step 2: Create dataset entity
    dataset = Dataset(
        name="Sample 2D Data",
        data=data,
        metadata={
            "description": "2D normal distribution with 4 anomalies",
            "features": ["feature1", "feature2"],
        },
    )

    # Save dataset
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)
    print(f"   Dataset saved with ID: {dataset.id[:8]}...")

    # Step 3: Create detector
    print("\n2. Creating Isolation Forest detector...")
    detector = Detector(
        name="Basic Isolation Forest",
        algorithm="IsolationForest",
        parameters={
            "contamination": 0.05,  # Expect 5% anomalies
            "n_estimators": 100,
            "random_state": 42,
        },
    )

    # Save detector
    detector_repo = container.detector_repository()
    detector_repo.save(detector)
    print(f"   Detector created with ID: {detector.id[:8]}...")

    # Step 4: Train detector
    print("\n3. Training detector...")
    detection_service = container.detection_service()

    # Train the model
    await detection_service.train_detector(detector_id=detector.id, dataset=dataset)
    print("   Training completed!")

    # Step 5: Detect anomalies
    print("\n4. Detecting anomalies...")
    result = await detection_service.detect_anomalies(
        detector_id=detector.id, dataset=dataset
    )

    # Step 6: Display results
    print("\n5. Results:")
    print(f"   Total samples: {result.total_samples}")
    print(f"   Anomalies found: {result.n_anomalies}")
    print(f"   Anomaly rate: {result.anomaly_rate:.1%}")
    print(f"   Anomaly indices: {result.anomaly_indices}")

    # Show which points were flagged as anomalies
    if result.n_anomalies > 0:
        print("\n   Anomalous data points:")
        anomalous_points = data.iloc[result.anomaly_indices]
        for idx, (_, row) in enumerate(anomalous_points.iterrows()):
            print(
                f"   [{result.anomaly_indices[idx]}] feature1={row['feature1']:.2f}, feature2={row['feature2']:.2f}"
            )

    # Optional: Save results
    result_repo = container.detection_result_repository()
    result_repo.save(result)
    print(f"\n   Results saved with ID: {result.id[:8]}...")

    print("\n✅ Basic anomaly detection completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
