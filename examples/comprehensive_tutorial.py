#!/usr/bin/env python3
"""
Pynomaly Comprehensive Tutorial
===============================

This tutorial demonstrates all major features of Pynomaly in a progressive manner,
from basic usage to advanced features like AutoML and explainable AI.

Tutorial Structure:
1. Basic Anomaly Detection
2. Multiple Algorithms Comparison
3. AutoML for Algorithm Selection
4. Explainable AI Analysis
5. Real-time Monitoring
6. Production Considerations
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


def print_section(title, level=1):
    """Print formatted section header."""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*40}")
        print(f"📊 {title}")
        print(f"{'-'*40}")


def create_tutorial_dataset():
    """Create a comprehensive dataset for the tutorial."""
    print("📁 Creating Tutorial Dataset...")

    np.random.seed(42)

    # Normal data (80% of samples)
    n_normal = 800

    # Multiple normal patterns
    pattern1 = np.random.multivariate_normal(
        [2, 2, 1], [[1, 0.5, 0], [0.5, 1, 0.2], [0, 0.2, 0.8]], n_normal // 3
    )
    pattern2 = np.random.multivariate_normal(
        [-1, -1, 0],
        [[0.8, -0.3, 0.1], [-0.3, 0.8, -0.1], [0.1, -0.1, 0.6]],
        n_normal // 3,
    )
    pattern3 = np.random.multivariate_normal(
        [0, 3, -1],
        [[1.2, 0.2, 0.3], [0.2, 1.2, -0.2], [0.3, -0.2, 1.0]],
        n_normal // 3 + n_normal % 3,
    )

    normal_data = np.vstack([pattern1, pattern2, pattern3])

    # Anomalous data (20% of samples)
    n_anomalies = 200

    # Different types of anomalies
    outliers = np.random.uniform(-6, 6, (n_anomalies // 2, 3))  # Random outliers
    cluster_outliers = np.random.multivariate_normal(
        [5, -5, 3], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], n_anomalies // 2
    )

    anomaly_data = np.vstack([outliers, cluster_outliers])

    # Combine and shuffle
    all_data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    labels = labels[indices]

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=["sensor_a", "sensor_b", "sensor_c"])

    dataset = Dataset(
        name="Tutorial Dataset",
        data=df,
        description="Multi-pattern dataset with various types of anomalies",
    )

    print(f"   ✅ Created dataset: {len(df)} samples, {len(df.columns)} features")
    print(
        f"   📊 Normal samples: {np.sum(labels == 0)}, Anomalous: {np.sum(labels == 1)}"
    )

    return dataset, labels


def tutorial_01_basic_detection(dataset, true_labels):
    """Tutorial 1: Basic Anomaly Detection."""
    print_section("Tutorial 1: Basic Anomaly Detection")

    print("🎯 Learning Objective: Understand basic anomaly detection with Pynomaly")
    print("📝 We'll use Isolation Forest - a popular tree-based algorithm")

    # Create detector
    print("\n🔧 Step 1: Create Isolation Forest detector")
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="Tutorial Isolation Forest",
        contamination_rate=ContaminationRate(0.2),  # Expect 20% anomalies
        random_state=42,
        n_estimators=100,
    )
    print(f"   ✅ Created detector: {detector.name}")
    print(f"   🎛️ Algorithm: {detector.algorithm_name}")
    print(f"   📊 Contamination rate: {detector.contamination_rate.value}")

    # Train detector
    print("\n🏋️ Step 2: Train the detector")
    start_time = time.time()
    detector.fit(dataset)
    training_time = time.time() - start_time

    print(f"   ✅ Training completed in {training_time:.3f}s")
    print(
        f"   📈 Trained on {dataset.n_samples} samples with {dataset.n_features} features"
    )

    # Run detection
    print("\n🔍 Step 3: Detect anomalies")
    result = detector.detect(dataset)

    print(f"   ✅ Detection completed in {result.execution_time_ms:.2f}ms")
    print(f"   🎯 Anomalies detected: {len(result.anomalies)}")
    print(f"   📊 Detection rate: {len(result.anomalies)/len(dataset.data)*100:.1f}%")
    print(f"   🎚️ Detection threshold: {result.threshold:.3f}")

    # Evaluate performance
    print("\n📈 Step 4: Evaluate performance")
    predicted_labels = result.labels
    accuracy = np.mean(predicted_labels == true_labels)

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

    print(f"   🎯 Accuracy: {accuracy:.3f}")
    print(f"   🎯 Precision: {precision:.3f}")
    print(f"   🎯 Recall: {recall:.3f}")
    print(f"   🎯 F1-Score: {f1_score:.3f}")

    # Show top anomalies
    print("\n🔍 Step 5: Analyze top anomalies")
    scores = np.array([score.value for score in result.scores])
    top_indices = np.argsort(scores)[-3:][::-1]

    for i, idx in enumerate(top_indices, 1):
        score = scores[idx]
        is_true_anomaly = true_labels[idx] == 1
        sample_values = dataset.data.iloc[idx].values

        print(
            f"   {i}. Sample {idx}: Score={score:.3f}, True anomaly={is_true_anomaly}, "
            f"Values=[{', '.join(f'{v:.2f}' for v in sample_values)}]"
        )

    print("\n✅ Tutorial 1 Complete! Key takeaways:")
    print("   - Isolation Forest is effective for high-dimensional anomaly detection")
    print("   - Contamination rate should match expected anomaly percentage")
    print("   - Higher anomaly scores indicate more unusual samples")

    return detector


def tutorial_02_algorithm_comparison(dataset, true_labels):
    """Tutorial 2: Multiple Algorithms Comparison."""
    print_section("Tutorial 2: Algorithm Comparison")

    print("🎯 Learning Objective: Compare different anomaly detection algorithms")
    print("📝 We'll test multiple algorithms and compare their performance")

    # Define algorithms to compare
    algorithms = [
        (SklearnAdapter, "IsolationForest", {"n_estimators": 100}),
        (SklearnAdapter, "LocalOutlierFactor", {}),
        (SklearnAdapter, "OneClassSVM", {"kernel": "rbf"}),
        (PyODAdapter, "LOF", {}),
        (PyODAdapter, "COPOD", {}),
        (PyODAdapter, "ECOD", {}),
    ]

    print(f"\n🧪 Testing {len(algorithms)} different algorithms...")

    results = []

    for adapter_class, algo_name, params in algorithms:
        print(f"\n   🔬 Testing {algo_name}...")

        try:
            # Create detector
            detector = adapter_class(
                algorithm_name=algo_name,
                contamination_rate=ContaminationRate(0.2),
                **params,
            )

            # Train and detect
            start_time = time.time()
            detector.fit(dataset)
            result = detector.detect(dataset)
            total_time = time.time() - start_time

            # Calculate metrics
            predicted_labels = result.labels
            accuracy = np.mean(predicted_labels == true_labels)

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

            result_data = {
                "algorithm": algo_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "execution_time": total_time,
                "anomalies_detected": len(result.anomalies),
            }

            results.append(result_data)

            print(
                f"      ✅ Accuracy: {accuracy:.3f}, F1: {f1_score:.3f}, Time: {total_time:.3f}s"
            )

        except Exception as e:
            print(f"      ❌ Failed: {str(e)[:50]}...")

    # Display comparison
    print("\n📊 Algorithm Comparison Results:")
    print(
        f"{'Algorithm':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9} {'Time (s)':<9}"
    )
    print("-" * 75)

    for result in sorted(results, key=lambda x: x["f1_score"], reverse=True):
        print(
            f"{result['algorithm']:<18} {result['accuracy']:<10.3f} {result['precision']:<11.3f} "
            f"{result['recall']:<8.3f} {result['f1_score']:<9.3f} {result['execution_time']:<9.3f}"
        )

    # Find best algorithm
    best_result = max(results, key=lambda x: x["f1_score"])
    print(f"\n🏆 Best performing algorithm: {best_result['algorithm']}")
    print(f"   🎯 F1-Score: {best_result['f1_score']:.3f}")
    print(f"   ⚡ Execution time: {best_result['execution_time']:.3f}s")

    print("\n✅ Tutorial 2 Complete! Key takeaways:")
    print("   - Different algorithms work better for different data patterns")
    print("   - Performance varies significantly across algorithms")
    print("   - Consider both accuracy and speed for production use")

    return results


def tutorial_03_automl_selection(dataset, true_labels):
    """Tutorial 3: AutoML for Algorithm Selection."""
    print_section("Tutorial 3: AutoML Algorithm Selection")

    print(
        "🎯 Learning Objective: Use AutoML to automatically select the best algorithm"
    )
    print("📝 AutoML will test multiple algorithms and hyperparameters automatically")

    # Simple AutoML implementation (inline for this tutorial)
    sys.path.append(str(Path(__file__).parent))
    try:
        from automl_example import SimpleAutoML
    except ImportError:
        print("   ⚠️ AutoML example not available, skipping detailed AutoML...")

    print("\n🤖 Running AutoML optimization...")
    try:
        automl = SimpleAutoML()
        summary = automl.run_automl(dataset, max_algorithms=8)
    except NameError:
        print(
            "   ⚠️ AutoML functionality requires automl_example.py - using simplified version"
        )
        summary = {"error": "AutoML example not available"}

    if "error" not in summary:
        print("\n🎯 AutoML Results:")
        print(f"   🧪 Algorithms tested: {summary['total_tested']}")
        print(f"   ✅ Successful runs: {summary['successful']}")
        print(f"   🏆 Best algorithm: {summary['best_algorithm']}")
        print(f"   📊 Best score: {summary['best_score']:.3f}")
        print(f"   ⚡ Total time: {summary['total_time']:.2f}s")

        print("\n🏆 Top 3 AutoML Recommendations:")
        for i, result in enumerate(summary["top_3"], 1):
            print(
                f"   {i}. {result['algorithm']:15} | Score: {result['avg_score']:.3f} | "
                f"Time: {result['execution_time']:.2f}s"
            )

    print("\n✅ Tutorial 3 Complete! Key takeaways:")
    print("   - AutoML can save time in algorithm selection")
    print("   - Automated testing reveals best performers")
    print("   - Consider both performance and computational cost")


def tutorial_04_explainable_ai(dataset, true_labels):
    """Tutorial 4: Explainable AI Analysis."""
    print_section("Tutorial 4: Explainable AI")

    print("🎯 Learning Objective: Understand why algorithms make specific decisions")
    print("📝 We'll analyze feature importance and explain individual predictions")

    # Use the explainer from our earlier example
    try:
        from explainable_ai_example import SimpleExplainer
    except ImportError:
        print(
            "   ⚠️ Explainable AI example not available, skipping detailed explanations..."
        )
        return

    print("\n🔧 Setting up explainable AI analysis...")

    # Create detector
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.2),
        random_state=42,
    )

    detector.fit(dataset)
    explainer = SimpleExplainer(detector)

    # Feature importance
    print("\n🧠 Analyzing feature importance...")
    importance_scores = explainer.calculate_feature_importance(dataset)

    print("\n📊 Feature Importance Results:")
    for name, importance in zip(dataset.feature_names, importance_scores, strict=False):
        bar = "█" * int(importance * 20)
        print(f"   {name:12}: {importance:.3f} {bar}")

    # Global explanation
    global_explanation = explainer.generate_global_explanation(dataset)

    print("\n🌍 Global Model Explanation:")
    print(f"   📊 Total samples: {global_explanation['total_samples']}")
    print(f"   🎯 Anomalies detected: {global_explanation['anomalies_detected']}")
    print(f"   📈 Detection rate: {global_explanation['detection_rate']:.1%}")

    # Individual explanations
    print("\n🔍 Explaining top anomalies...")
    explanations = explainer.explain_prediction(dataset)

    for i, explanation in enumerate(explanations[:2], 1):
        print(f"\n   Anomaly #{i} (Sample {explanation['sample_index']}):")
        print(f"      Score: {explanation['anomaly_score']:.3f}")

        contributions = explanation["feature_contributions"]
        sorted_contributions = sorted(
            contributions.items(), key=lambda x: x[1]["contribution"], reverse=True
        )

        for feature, contrib in sorted_contributions[:2]:
            print(
                f"         {feature}: {contrib['value']:.2f} "
                f"(contribution: {contrib['contribution']:.3f})"
            )

    print("\n✅ Tutorial 4 Complete! Key takeaways:")
    print("   - Feature importance reveals which sensors matter most")
    print("   - Individual explanations show why samples are anomalous")
    print("   - Explainability builds trust in ML decisions")


def tutorial_05_production_considerations(dataset):
    """Tutorial 5: Production Considerations."""
    print_section("Tutorial 5: Production Considerations")

    print("🎯 Learning Objective: Understand production deployment considerations")
    print("📝 We'll cover performance, monitoring, and scalability aspects")

    print("\n⚡ Performance Testing:")

    # Test performance with different dataset sizes
    sizes = [100, 500, 1000]
    algorithms = ["IsolationForest", "LOF", "COPOD"]

    for size in sizes:
        print(f"\n   📊 Testing with {size} samples:")
        subset_data = dataset.data.iloc[:size]
        subset_dataset = Dataset(name=f"Subset_{size}", data=subset_data)

        for algo in algorithms:
            try:
                if algo == "IsolationForest":
                    detector = SklearnAdapter(
                        algorithm_name=algo, contamination_rate=ContaminationRate(0.1)
                    )
                else:
                    detector = PyODAdapter(
                        algorithm_name=algo, contamination_rate=ContaminationRate(0.1)
                    )

                start_time = time.time()
                detector.fit(subset_dataset)
                result = detector.detect(subset_dataset)
                total_time = time.time() - start_time

                throughput = size / total_time
                print(
                    f"      {algo:15}: {total_time:.3f}s ({throughput:.0f} samples/sec)"
                )

            except Exception as e:
                print(f"      {algo:15}: Error - {str(e)[:30]}...")

    print("\n🏗️ Production Deployment Checklist:")
    checklist = [
        "✅ Algorithm selected and validated",
        "✅ Performance benchmarks established",
        "✅ Error handling implemented",
        "✅ Monitoring and logging configured",
        "✅ Docker containers prepared",
        "✅ Kubernetes manifests ready",
        "✅ CI/CD pipeline configured",
        "✅ Health checks implemented",
    ]

    for item in checklist:
        print(f"   {item}")

    print("\n📊 Monitoring Recommendations:")
    monitoring = [
        "Track detection latency and throughput",
        "Monitor anomaly detection rates over time",
        "Alert on unusual detection patterns",
        "Log prediction confidence scores",
        "Monitor model drift and performance degradation",
    ]

    for i, item in enumerate(monitoring, 1):
        print(f"   {i}. {item}")

    print("\n✅ Tutorial 5 Complete! Key takeaways:")
    print("   - Performance testing is crucial for production")
    print("   - Monitoring ensures continued model effectiveness")
    print("   - Infrastructure setup enables scalable deployment")


def run_comprehensive_tutorial():
    """Run the complete Pynomaly tutorial."""
    print("🎓 Welcome to the Pynomaly Comprehensive Tutorial!")
    print("=" * 60)
    print("This tutorial will guide you through all major features of Pynomaly,")
    print("from basic anomaly detection to advanced AutoML and production deployment.")
    print("\nEstimated time: 5-10 minutes")

    # Create dataset
    dataset, true_labels = create_tutorial_dataset()

    # Run tutorials
    tutorial_01_basic_detection(dataset, true_labels)
    tutorial_02_algorithm_comparison(dataset, true_labels)
    tutorial_03_automl_selection(dataset, true_labels)
    tutorial_04_explainable_ai(dataset, true_labels)
    tutorial_05_production_considerations(dataset)

    # Final summary
    print_section("🎉 Tutorial Complete!")
    print("Congratulations! You've learned how to:")
    print("   ✅ Perform basic anomaly detection")
    print("   ✅ Compare multiple algorithms")
    print("   ✅ Use AutoML for algorithm selection")
    print("   ✅ Explain model decisions")
    print("   ✅ Consider production deployment")

    print("\n📚 Next Steps:")
    print("   🔗 Explore the web interface: python scripts/run/run_web_app.py")
    print(
        "   🔗 Try real-time monitoring: python examples/realtime_monitoring_example.py"
    )
    print("   🔗 Read the documentation: docs/")
    print("   🔗 Deploy to production using provided Docker/Kubernetes files")

    print("\n🚀 Ready for production anomaly detection with Pynomaly!")


def main():
    """Main function."""
    try:
        run_comprehensive_tutorial()
        return 0
    except Exception as e:
        print(f"❌ Tutorial failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
