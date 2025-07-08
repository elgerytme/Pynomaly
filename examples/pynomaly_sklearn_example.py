#!/usr/bin/env python3
"""
Pynomaly SklearnAdapter Example
==============================

This example demonstrates how to use Pynomaly's SklearnAdapter properly
with the clean architecture approach.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


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


def run_pynomaly_sklearn_example():
    """Run Pynomaly SklearnAdapter example."""
    print("🔍 Pynomaly SklearnAdapter Example")
    print("=" * 50)

    # Step 1: Create sample data
    data_df = create_sample_data(n_samples=1000, contamination=0.1)

    # Step 2: Create Pynomaly dataset (excluding true labels)
    feature_data = data_df[["feature_1", "feature_2"]]
    dataset = Dataset(
        name="Pynomaly Example Dataset",
        data=feature_data,
        feature_names=["feature_1", "feature_2"],
    )

    print(
        f"📊 Dataset created: {len(dataset.data)} samples, {len(dataset.feature_names)} features"
    )

    # Step 3: Create detector using Pynomaly SklearnAdapter
    try:
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="Pynomaly Isolation Forest",
            contamination_rate=ContaminationRate(0.1),
            random_state=42,
            n_estimators=100,
        )
        print(f"🤖 Created Pynomaly detector: {detector.name}")
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        return False

    # Step 4: Fit the detector
    try:
        print("🔧 Training detector...")
        detector.fit(dataset)
        print("✅ Detector trained successfully")
    except Exception as e:
        print(f"❌ Failed to train detector: {e}")
        return False

    # Step 5: Detect anomalies
    try:
        print("🔍 Detecting anomalies...")
        result = detector.detect(dataset)
        print("✅ Anomaly detection completed")
    except Exception as e:
        print(f"❌ Failed to detect anomalies: {e}")
        return False

    # Step 6: Analyze results
    print("\n📈 Results Analysis")
    print("-" * 30)

    n_detected_anomalies = len(result.anomalies)
    n_true_anomalies = np.sum(data_df["true_label"])

    print(f"Total samples: {len(dataset.data)}")
    print(f"True anomalies: {n_true_anomalies}")
    print(f"Detected anomalies: {n_detected_anomalies}")
    print(f"Detection rate: {n_detected_anomalies/len(dataset.data)*100:.1f}%")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"Detection threshold: {result.threshold:.3f}")

    # Calculate accuracy metrics (since we have true labels)
    true_labels = data_df["true_label"].values
    predicted_labels = result.labels
    accuracy = np.mean(predicted_labels == true_labels)

    # Calculate precision and recall
    true_positives = np.sum((predicted_labels == 1) & (true_labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (true_labels == 1))

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
    scores = [score.value for score in result.scores]
    print("\n📊 Anomaly Scores:")
    print(f"Min score: {min(scores):.3f}")
    print(f"Max score: {max(scores):.3f}")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Std score: {np.std(scores):.3f}")

    # Show detector metadata
    print("\n🔧 Detector Information:")
    print(f"Algorithm: {detector.algorithm_name}")
    print(f"Fitted: {detector.is_fitted}")
    print(
        f"Training time: {detector.metadata.get('training_time_ms', 'unknown'):.2f}ms"
    )
    print(f"Training samples: {detector.metadata.get('training_samples', 'unknown')}")
    print(f"Training features: {detector.metadata.get('training_features', 'unknown')}")

    print("\n🎉 Pynomaly SklearnAdapter example completed successfully!")
    print("\nNext steps:")
    print(
        "- Try different algorithms: LocalOutlierFactor, OneClassSVM, EllipticEnvelope"
    )
    print("- Experiment with different contamination rates")
    print("- Use your own data by replacing the create_sample_data() function")
    print("- Explore the web interface: python scripts/run/run_web_app.py")

    return True


def main():
    """Main function."""
    try:
        success = run_pynomaly_sklearn_example()
        return 0 if success else 1
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Solution: Install Pynomaly with:")
        print("   pip install -e .")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
