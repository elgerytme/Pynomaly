#!/usr/bin/env python3
"""
Pynomaly Quick Demo
==================

A 2-minute demonstration of Pynomaly's core capabilities.
Perfect for showcasing the library's power and simplicity.
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


def create_demo_data():
    """Create interesting demo data with clear anomaly patterns."""
    np.random.seed(42)

    print("🎲 Generating synthetic IoT sensor data...")

    # Normal operations: server metrics with typical patterns
    n_normal = 950

    # Normal CPU usage (20-60%)
    cpu_normal = np.random.beta(2, 3, n_normal) * 40 + 20

    # Normal memory usage (correlated with CPU)
    memory_normal = cpu_normal * 0.8 + np.random.normal(0, 5, n_normal)
    memory_normal = np.clip(memory_normal, 10, 90)

    # Normal network traffic (independent)
    network_normal = np.random.exponential(30, n_normal)
    network_normal = np.clip(network_normal, 0, 100)

    # Anomalies: system failures and attacks
    n_anomalies = 50

    # CPU spikes (DDoS attack)
    cpu_anomaly = np.random.uniform(90, 100, n_anomalies)

    # Memory leaks (memory usage spikes)
    memory_anomaly = np.random.uniform(85, 100, n_anomalies)

    # Network anomalies (unusual traffic)
    network_anomaly = np.random.choice([0, 200], n_anomalies) + np.random.normal(
        0, 10, n_anomalies
    )
    network_anomaly = np.clip(network_anomaly, 0, 300)

    # Combine data
    cpu = np.concatenate([cpu_normal, cpu_anomaly])
    memory = np.concatenate([memory_normal, memory_anomaly])
    network = np.concatenate([network_normal, network_anomaly])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle
    indices = np.random.permutation(len(cpu))

    df = pd.DataFrame(
        {
            "cpu_usage_percent": cpu[indices],
            "memory_usage_percent": memory[indices],
            "network_traffic_mbps": network[indices],
        }
    )

    dataset = Dataset(
        name="IoT Server Monitoring Data",
        data=df,
        description="Simulated server metrics with normal operations and cyber attacks",
    )

    print(
        f"   📊 Created: {len(df)} samples ({n_normal} normal, {n_anomalies} anomalies)"
    )
    print("   🖥️ Metrics: CPU usage, Memory usage, Network traffic")

    return dataset, labels[indices]


def demo_basic_detection(dataset, true_labels):
    """Demonstrate basic anomaly detection."""
    print("\n🔍 DEMO: Basic Anomaly Detection")
    print("-" * 40)

    # Create detector
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="IoT Security Monitor",
        contamination_rate=ContaminationRate(0.05),  # Expect 5% anomalies
        random_state=42,
    )

    print(f"🤖 Using {detector.algorithm_name} for anomaly detection...")

    # Train and detect
    start_time = time.time()
    detector.fit(dataset)
    result = detector.detect(dataset)
    total_time = time.time() - start_time

    # Results
    accuracy = np.mean(result.labels == true_labels)
    anomalies_found = np.sum(result.labels)

    print(f"   ⚡ Analysis completed in {total_time:.3f} seconds")
    print(f"   🎯 Accuracy: {accuracy:.1%}")
    print(f"   🚨 Anomalies detected: {anomalies_found} out of {len(dataset.data)}")
    print(f"   📊 Detection rate: {anomalies_found/len(dataset.data)*100:.1f}%")

    # Show top threats
    scores = np.array([score.value for score in result.scores])
    threat_indices = np.where(result.labels == 1)[0]
    top_threats = threat_indices[np.argsort(scores[threat_indices])[-3:]][::-1]

    print("\n🚨 Top 3 Security Threats Detected:")
    for i, idx in enumerate(top_threats, 1):
        sample = dataset.data.iloc[idx]
        threat_score = scores[idx]
        is_real_threat = true_labels[idx] == 1

        print(
            f"   {i}. Sample {idx}: Score={threat_score:.3f} ({'✓ Real threat' if is_real_threat else '✗ False alarm'})"
        )
        print(
            f"      CPU: {sample['cpu_usage_percent']:.1f}%, "
            f"Memory: {sample['memory_usage_percent']:.1f}%, "
            f"Network: {sample['network_traffic_mbps']:.1f} Mbps"
        )


def demo_algorithm_speed_test(dataset):
    """Demonstrate algorithm speed comparison."""
    print("\n⚡ DEMO: Real-time Performance Test")
    print("-" * 40)

    algorithms = [
        (SklearnAdapter, "IsolationForest"),
        (PyODAdapter, "LOF"),
        (PyODAdapter, "COPOD"),
        (PyODAdapter, "ECOD"),
    ]

    print("🏃 Testing real-time detection speed...")

    results = []
    for adapter_class, algo_name in algorithms:
        try:
            detector = adapter_class(
                algorithm_name=algo_name, contamination_rate=ContaminationRate(0.05)
            )

            # Train
            train_start = time.time()
            detector.fit(dataset)
            train_time = time.time() - train_start

            # Detect
            detect_start = time.time()
            result = detector.detect(dataset)
            detect_time = time.time() - detect_start

            throughput = len(dataset.data) / detect_time

            results.append(
                {
                    "algorithm": algo_name,
                    "train_time": train_time,
                    "detect_time": detect_time,
                    "throughput": throughput,
                    "anomalies": len(result.anomalies),
                }
            )

        except Exception as e:
            print(f"   ❌ {algo_name}: {str(e)[:30]}...")

    # Display results
    print(f"\n📊 Performance Results (for {len(dataset.data)} samples):")
    print(
        f"{'Algorithm':<15} {'Train (s)':<10} {'Detect (s)':<11} {'Throughput':<12} {'Anomalies':<10}"
    )
    print("-" * 65)

    for result in sorted(results, key=lambda x: x["throughput"], reverse=True):
        print(
            f"{result['algorithm']:<15} {result['train_time']:<10.3f} "
            f"{result['detect_time']:<11.3f} {result['throughput']:<12.0f} {result['anomalies']:<10}"
        )

    fastest = max(results, key=lambda x: x["throughput"])
    print(
        f"\n🏆 Fastest: {fastest['algorithm']} at {fastest['throughput']:.0f} samples/sec"
    )


def demo_feature_importance(dataset):
    """Demonstrate simple feature importance analysis."""
    print("\n🧠 DEMO: AI Explainability")
    print("-" * 40)

    print("🔍 Analyzing which metrics matter most for threat detection...")

    # Train detector
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.05),
        random_state=42,
    )
    detector.fit(dataset)

    # Simple feature importance via permutation
    baseline_result = detector.detect(dataset)
    baseline_scores = np.array([score.value for score in baseline_result.scores])
    baseline_mean = np.mean(baseline_scores)

    feature_importance = {}

    for feature in dataset.feature_names:
        # Permute feature
        permuted_data = dataset.data.copy()
        permuted_data[feature] = np.random.permutation(permuted_data[feature].values)

        permuted_dataset = Dataset(name="Permuted", data=permuted_data)
        permuted_result = detector.detect(permuted_dataset)
        permuted_scores = np.array([score.value for score in permuted_result.scores])
        permuted_mean = np.mean(permuted_scores)

        importance = abs(baseline_mean - permuted_mean)
        feature_importance[feature] = importance

    # Normalize and display
    total_importance = sum(feature_importance.values())

    print("\n📊 Feature Importance for Threat Detection:")
    for feature, importance in sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        normalized_importance = (
            importance / total_importance if total_importance > 0 else 0
        )
        bar = "█" * int(normalized_importance * 30)
        print(f"   {feature:<25}: {normalized_importance:.1%} {bar}")

    most_important = max(feature_importance.items(), key=lambda x: x[1])
    print(f"\n🎯 Most critical metric: {most_important[0].replace('_', ' ').title()}")


def run_quick_demo():
    """Run the complete quick demo."""
    print("🚀 Pynomaly Quick Demo")
    print("=" * 50)
    print("🎯 Demonstrating AI-powered cybersecurity monitoring")
    print("⏱️  Duration: ~2 minutes")

    # Generate demo data
    dataset, true_labels = create_demo_data()

    # Run demonstrations
    demo_basic_detection(dataset, true_labels)
    demo_algorithm_speed_test(dataset)
    demo_feature_importance(dataset)

    # Summary
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("✨ What you just saw:")
    print("   🔍 Real-time anomaly detection with 90%+ accuracy")
    print("   ⚡ Sub-second processing of 1000+ data points")
    print("   🧠 AI explainability showing which metrics matter")
    print("   🛡️ Automated cybersecurity threat detection")

    print("\n🚀 Next Steps:")
    print("   📖 Run the full tutorial: python examples/comprehensive_tutorial.py")
    print("   🌐 Try the web interface: python scripts/run/run_web_app.py")
    print("   🤖 Explore AutoML: python examples/automl_example.py")
    print("   📊 Monitor in real-time: python examples/realtime_monitoring_example.py")

    print(
        "\n💡 Pynomaly: Production-ready anomaly detection in just a few lines of code!"
    )


def main():
    """Main function."""
    try:
        run_quick_demo()
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
