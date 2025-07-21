#!/usr/bin/env python3
"""
Getting Started with Pynomaly v0.1.1

This example demonstrates the basic usage of Pynomaly for anomaly detection.
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import sys
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interfaces.domain.entities.dataset import Dataset
from interfaces.domain.value_objects.contamination_rate import ContaminationRate
# TODO: Create local sklearn adapter


def generate_sample_data():
    """Generate sample data with some anomalies."""
    np.random.seed(42)

    # Normal data
    normal_data = np.random.normal(0, 1, (800, 5))

    # Anomalous data
    anomalous_data = np.random.normal(3, 0.5, (200, 5))

    # Combine data
    data = np.vstack([normal_data, anomalous_data])

    # Create labels (1 for anomaly, 0 for normal)
    labels = np.hstack([np.zeros(800), np.ones(200)])

    return data, labels


def basic_anomaly_detection():
    """Demonstrate basic anomaly detection with Pynomaly."""
    print("🔍 Basic Anomaly Detection with Pynomaly")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating sample data...")
    data, true_labels = generate_sample_data()
    print(f"   Dataset shape: {data.shape}")
    print(f"   True anomalies: {np.sum(true_labels)}")

    # Create dataset
    dataset = Dataset(name="sample_data", data=data)
    print(f"   Created dataset: {dataset.name}")

    # Create detector
    print("\n🤖 Creating anomaly detector...")
    detector = SklearnAdapter(
        "IsolationForest", contamination_rate=ContaminationRate(0.2)
    )
    print(f"   Detector: {detector.name}")
    print(f"   Algorithm: {detector.algorithm_name}")

    # Fit detector
    print("\n🔧 Training detector...")
    detector.fit(dataset)
    print("   ✅ Detector trained successfully")

    # Detect anomalies
    print("\n🎯 Detecting anomalies...")
    result = detector.detect(dataset)
    print(f"   Detected {len(result.anomalies)} anomalies")
    print(f"   Threshold: {result.threshold:.4f}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")

    # Calculate basic metrics
    predicted_labels = result.labels
    true_positives = np.sum((predicted_labels == 1) & (true_labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))
    true_negatives = np.sum((predicted_labels == 0) & (true_labels == 0))
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

    print("\n📈 Performance Metrics:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   True Positives: {true_positives}")
    print(f"   False Positives: {false_positives}")

    return result


def compare_algorithms():
    """Compare different anomaly detection algorithms."""
    print("\n🔬 Comparing Anomaly Detection Algorithms")
    print("=" * 50)

    # Generate data
    data, true_labels = generate_sample_data()
    dataset = Dataset(name="comparison_data", data=data)

    # Algorithms to compare
    algorithms = [
        "IsolationForest",
        "OneClassSVM",
        "LocalOutlierFactor",
        "EllipticEnvelope",
    ]

    results = {}

    for algorithm in algorithms:
        print(f"\n🔍 Testing {algorithm}...")

        try:
            # Create and train detector
            detector = SklearnAdapter(
                algorithm, contamination_rate=ContaminationRate(0.2)
            )
            detector.fit(dataset)

            # Detect anomalies
            result = detector.detect(dataset)

            # Calculate metrics
            predicted_labels = result.labels
            true_positives = np.sum((predicted_labels == 1) & (true_labels == 1))
            false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / np.sum(true_labels) if np.sum(true_labels) > 0 else 0
            )
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results[algorithm] = {
                "anomalies_detected": len(result.anomalies),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "execution_time_ms": result.execution_time_ms,
            }

            print(
                f"   ✅ {algorithm}: {len(result.anomalies)} anomalies, F1={f1_score:.3f}"
            )

        except Exception as e:
            print(f"   ❌ {algorithm}: Failed - {e}")
            results[algorithm] = {"error": str(e)}

    # Display comparison
    print("\n📊 Algorithm Comparison:")
    print(
        f"{'Algorithm':<20} {'Anomalies':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time(ms)':<10}"
    )
    print("-" * 70)

    for algorithm, metrics in results.items():
        if "error" not in metrics:
            print(
                f"{algorithm:<20} {metrics['anomalies_detected']:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['execution_time_ms']:<10.2f}"
            )
        else:
            print(f"{algorithm:<20} {'ERROR':<10} {metrics['error'][:40]:<40}")

    return results


def advanced_usage():
    """Demonstrate advanced usage patterns."""
    print("\n⚙️ Advanced Usage Patterns")
    print("=" * 50)

    # Generate data
    data, _ = generate_sample_data()
    dataset = Dataset(name="advanced_data", data=data)

    # Custom parameters
    print("🔧 Custom Parameters:")
    detector = SklearnAdapter(
        "IsolationForest",
        contamination_rate=ContaminationRate(0.15),
        n_estimators=200,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=42,
    )

    # Display parameters
    params = detector.get_params()
    print("   Current Parameters:")
    for key, value in params.items():
        print(f"     {key}: {value}")

    # Fit and detect
    detector.fit(dataset)
    result = detector.detect(dataset)

    print("\n   Results with custom parameters:")
    print(f"     Anomalies detected: {len(result.anomalies)}")
    print(f"     Threshold: {result.threshold:.4f}")

    # Parameter modification
    print("\n🔄 Parameter Modification:")
    detector.set_params(contamination=0.25, n_estimators=100)
    detector.fit(dataset)
    result2 = detector.detect(dataset)

    print("   Results after parameter change:")
    print(f"     Anomalies detected: {len(result2.anomalies)}")
    print(f"     Threshold: {result2.threshold:.4f}")

    # Score calculation
    print("\n📊 Score Calculation:")
    scores = detector.score(dataset)
    print(f"   Total scores: {len(scores)}")
    print(
        f"   Score range: {min(s.value for s in scores):.4f} - {max(s.value for s in scores):.4f}"
    )
    print(f"   Average confidence: {np.mean([s.confidence for s in scores]):.4f}")


def error_handling_examples():
    """Demonstrate error handling."""
    print("\n🚨 Error Handling Examples")
    print("=" * 50)

    # Invalid algorithm
    print("❌ Invalid Algorithm:")
    try:
        detector = SklearnAdapter("InvalidAlgorithm")
        print("   This should not print")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {type(e).__name__}: {e}")

    # Prediction without fitting
    print("\n❌ Prediction Without Fitting:")
    try:
        detector = SklearnAdapter("IsolationForest")
        data, _ = generate_sample_data()
        dataset = Dataset(name="test", data=data)
        detector.predict(dataset)
        print("   This should not print")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {type(e).__name__}: {e}")

    # Empty dataset
    print("\n❌ Empty Dataset:")
    try:
        detector = SklearnAdapter("IsolationForest")
        empty_data = np.array([]).reshape(0, 5)
        empty_dataset = Dataset(name="empty", data=empty_data)
        detector.fit(empty_dataset)
        print("   This should not print")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {type(e).__name__}: {e}")


def main():
    """Main function to run all examples."""
    print("🚀 Pynomaly v0.1.1 - Getting Started Examples")
    print("=" * 60)

    try:
        # Basic usage
        basic_anomaly_detection()

        # Algorithm comparison
        compare_algorithms()

        # Advanced usage
        advanced_usage()

        # Error handling
        error_handling_examples()

        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("- Try with your own data")
        print("- Explore different algorithms")
        print("- Adjust parameters for better performance")
        print("- Check out the documentation for more features")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
