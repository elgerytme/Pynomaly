#!/usr/bin/env python3
"""
Quick Start Example for Pynomaly
================================

This example demonstrates the basic functionality of Pynomaly with a simple,
working anomaly detection scenario using synthetic data.

Requirements:
- Only core dependencies (no optional packages required)
- Works with basic installation: pip install -e .
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from pynomaly_detection.domain.entities import Dataset


def create_sample_data(n_samples=1000, contamination=0.1):
    """Create sample data with known anomalies."""
    print(f"Creating sample dataset with {n_samples} samples...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate normal data
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal

    # Normal data: two clusters
    cluster1 = np.random.normal([2, 2], [0.5, 0.5], (n_normal // 2, 2))
    cluster2 = np.random.normal([-2, -2], [0.5, 0.5], (n_normal - n_normal // 2, 2))
    normal_data = np.vstack([cluster1, cluster2])

    # Anomalous data: scattered points
    anomalies = np.random.uniform(-6, 6, (n_anomalies, 2))

    # Combine data
    data = np.vstack([normal_data, anomalies])

    # Create labels (1 for anomaly, 0 for normal) - for evaluation only
    labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle the data
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["feature_1", "feature_2"])
    df["true_label"] = labels  # Include for evaluation

    print(f"✅ Created dataset: {n_normal} normal samples, {n_anomalies} anomalies")
    return df


def run_basic_detection():
    """Run basic anomaly detection example."""
    print("🔍 Pynomaly Quick Start Example")
    print("=" * 50)

    # Step 1: Create sample data
    data_df = create_sample_data(n_samples=1000, contamination=0.1)

    # Step 2: Create Pynomaly dataset (excluding true labels)
    feature_data = data_df[["feature_1", "feature_2"]]
    dataset = Dataset(
        name="Quick Start Dataset",
        data=feature_data,
        feature_names=["feature_1", "feature_2"],
    )

    print(
        f"📊 Dataset created: {len(dataset.data)} samples, {len(dataset.feature_names)} features"
    )

    # Step 3: Create detector using Isolation Forest (direct sklearn usage)
    try:
        from sklearn.ensemble import IsolationForest

        detector = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        print("🤖 Created Isolation Forest detector")
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        return False

    # Step 4: Fit the detector
    try:
        print("🔧 Training detector...")
        detector.fit(dataset.data.values)
        print("✅ Detector trained successfully")
    except Exception as e:
        print(f"❌ Failed to train detector: {e}")
        return False

    # Step 5: Make predictions
    try:
        print("🔍 Detecting anomalies...")
        predictions = detector.predict(dataset.data.values)  # 1 = normal, -1 = anomaly
        scores = detector.decision_function(dataset.data.values)  # Anomaly scores

        # Convert predictions to binary (1 = anomaly, 0 = normal)
        # sklearn returns 1 for normal, -1 for anomaly, so we need to flip
        binary_predictions = (predictions == -1).astype(int)

        print("✅ Anomaly detection completed")
    except Exception as e:
        print(f"❌ Failed to detect anomalies: {e}")
        return False

    # Step 6: Analyze results
    print("\n📈 Results Analysis")
    print("-" * 30)

    n_detected_anomalies = np.sum(binary_predictions)
    n_true_anomalies = np.sum(data_df["true_label"])

    print(f"Total samples: {len(dataset.data)}")
    print(f"True anomalies: {n_true_anomalies}")
    print(f"Detected anomalies: {n_detected_anomalies}")
    print(f"Detection rate: {n_detected_anomalies/len(dataset.data)*100:.1f}%")

    # Calculate accuracy metrics (since we have true labels)
    true_labels = data_df["true_label"].values
    accuracy = np.mean(binary_predictions == true_labels)

    # Calculate precision and recall
    true_positives = np.sum((binary_predictions == 1) & (true_labels == 1))
    false_positives = np.sum((binary_predictions == 1) & (true_labels == 0))
    false_negatives = np.sum((binary_predictions == 0) & (true_labels == 1))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n🎯 Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")

    # Show score statistics
    print("\n📊 Anomaly Scores:")
    print(f"Min score: {scores.min():.3f}")
    print(f"Max score: {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Std score: {scores.std():.3f}")

    print("\n🎉 Quick start example completed successfully!")
    print("\nNext steps:")
    print("- Try different algorithms: LocalOutlierFactor, OneClassSVM")
    print("- Experiment with different contamination rates")
    print("- Use your own data by replacing the create_sample_data() function")
    print("- Explore the web interface: python scripts/run/run_web_app.py")

    return True


def main():
    """Main function."""
    try:
        success = run_basic_detection()
        return 0 if success else 1
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Solution: Install Pynomaly with:")
        print("   pip install -e .")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
