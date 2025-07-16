#!/usr/bin/env python3
"""
Pynomaly Industry Templates
===========================

Industry-specific anomaly detection templates demonstrating
how to adapt Pynomaly for different sectors and use cases.

Templates:
1. Financial Fraud Detection
2. Industrial IoT Monitoring
3. Cybersecurity Threat Detection
4. Healthcare Patient Monitoring
5. Supply Chain Optimization
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


import numpy as np
import pandas as pd

from interfaces.domain.entities import Dataset
from interfaces.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class FinancialFraudDetector:
    """Financial fraud detection template."""

    def __init__(self):
        self.name = "Financial Fraud Detection"
        self.recommended_algorithms = ["IsolationForest", "LOF", "COPOD"]
        self.contamination_rate = 0.01  # 1% fraud rate

    def create_sample_data(self, n_samples=10000):
        """Create sample financial transaction data."""
        np.random.seed(42)

        print(f"🏦 Generating {n_samples} financial transactions...")

        # Normal transactions (99%)
        n_normal = int(n_samples * 0.99)

        # Transaction amounts (log-normal distribution)
        amounts_normal = np.random.lognormal(3, 1.5, n_normal)
        amounts_normal = np.clip(amounts_normal, 1, 10000)  # $1 to $10,000

        # Transaction frequency (per user per day)
        frequency_normal = np.random.poisson(3, n_normal)  # 3 transactions/day average

        # Account age (days)
        account_age_normal = np.random.exponential(365, n_normal)  # Average 1 year

        # Geographic diversity (0-1, higher = more locations)
        geo_diversity_normal = np.random.beta(2, 8, n_normal)  # Low diversity normal

        # Fraudulent transactions (1%)
        n_fraud = n_samples - n_normal

        # Fraud patterns: unusual amounts, high frequency, new accounts, high mobility
        fraud_type = np.random.choice([0, 1, 2], n_fraud)
        amounts_fraud = np.zeros(n_fraud)
        for i, ftype in enumerate(fraud_type):
            if ftype == 0:
                amounts_fraud[i] = np.random.uniform(10000, 50000)  # Large amounts
            elif ftype == 1:
                amounts_fraud[i] = np.random.uniform(0.01, 1)  # Micro transactions
            else:
                amounts_fraud[i] = np.random.lognormal(8, 1)  # Very large amounts

        frequency_fraud = np.random.poisson(15, n_fraud)  # High frequency
        account_age_fraud = np.random.exponential(30, n_fraud)  # New accounts
        geo_diversity_fraud = np.random.beta(8, 2, n_fraud)  # High mobility

        # Combine data
        amounts = np.concatenate([amounts_normal, amounts_fraud])
        frequencies = np.concatenate([frequency_normal, frequency_fraud])
        account_ages = np.concatenate([account_age_normal, account_age_fraud])
        geo_diversity = np.concatenate([geo_diversity_normal, geo_diversity_fraud])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

        # Shuffle
        indices = np.random.permutation(len(amounts))

        df = pd.DataFrame(
            {
                "transaction_amount": amounts[indices],
                "daily_frequency": frequencies[indices],
                "account_age_days": account_ages[indices],
                "geographic_diversity": geo_diversity[indices],
            }
        )

        dataset = Dataset(
            name="Financial Transactions",
            data=df,
            description="Credit card transactions with fraud indicators",
        )

        return dataset, labels[indices]

    def detect_fraud(self, dataset, true_labels):
        """Run fraud detection analysis."""
        print(f"\n🔍 {self.name} Analysis")
        print("-" * 50)

        # Test multiple algorithms
        results = []
        for algo in self.recommended_algorithms:
            try:
                if algo in ["IsolationForest"]:
                    detector = SklearnAdapter(
                        algorithm_name=algo,
                        contamination_rate=ContaminationRate(self.contamination_rate),
                    )
                else:
                    detector = PyODAdapter(
                        algorithm_name=algo,
                        contamination_rate=ContaminationRate(self.contamination_rate),
                    )

                detector.fit(dataset)
                result = detector.detect(dataset)

                accuracy = np.mean(result.labels == true_labels)
                precision = np.sum((result.labels == 1) & (true_labels == 1)) / max(
                    np.sum(result.labels == 1), 1
                )
                recall = np.sum((result.labels == 1) & (true_labels == 1)) / max(
                    np.sum(true_labels == 1), 1
                )

                results.append(
                    {
                        "algorithm": algo,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "detected_fraud": np.sum(result.labels == 1),
                        "execution_time": result.execution_time_ms / 1000,
                    }
                )

            except Exception as e:
                print(f"   ❌ {algo}: {str(e)[:50]}...")

        # Display results
        print("📊 Fraud Detection Results:")
        print(
            f"{'Algorithm':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'Detected':<9} {'Time (s)':<9}"
        )
        print("-" * 75)

        for result in sorted(results, key=lambda x: x["precision"], reverse=True):
            print(
                f"{result['algorithm']:<15} {result['accuracy']:<10.3f} {result['precision']:<11.3f} "
                f"{result['recall']:<8.3f} {result['detected_fraud']:<9} {result['execution_time']:<9.3f}"
            )

        best = max(results, key=lambda x: x["precision"])
        print(
            f"\n🏆 Best for fraud detection: {best['algorithm']} (Precision: {best['precision']:.3f})"
        )

        return results


class IndustrialIoTMonitor:
    """Industrial IoT monitoring template."""

    def __init__(self):
        self.name = "Industrial IoT Monitoring"
        self.recommended_algorithms = ["ECOD", "COPOD", "HBOS"]
        self.contamination_rate = 0.05  # 5% equipment anomalies

    def create_sample_data(self, n_samples=5000):
        """Create sample industrial sensor data."""
        np.random.seed(42)

        print(f"🏭 Generating {n_samples} IoT sensor readings...")

        # Normal operations (95%)
        n_normal = int(n_samples * 0.95)

        # Temperature (°C) - should be around 80°C ± 10°C
        temp_normal = np.random.normal(80, 5, n_normal)

        # Pressure (bar) - correlated with temperature
        pressure_normal = temp_normal * 0.5 + np.random.normal(40, 2, n_normal)

        # Vibration (mm/s) - low during normal operation
        vibration_normal = np.random.gamma(2, 2, n_normal)

        # Power consumption (kW) - correlated with operation
        power_normal = (temp_normal + pressure_normal) * 0.3 + np.random.normal(
            0, 5, n_normal
        )

        # Anomalies (5%) - equipment failures
        n_anomalies = n_samples - n_normal

        # Equipment failure patterns
        failure_type = np.random.choice([0, 1, 2], n_anomalies)
        temp_anomaly = np.zeros(n_anomalies)
        for i, ftype in enumerate(failure_type):
            if ftype == 0:
                temp_anomaly[i] = np.random.uniform(120, 150)  # Overheating
            elif ftype == 1:
                temp_anomaly[i] = np.random.uniform(30, 50)  # Cooling failure
            else:
                temp_anomaly[i] = np.random.normal(80, 20)  # Temperature instability

        pressure_anomaly = np.random.uniform(100, 200, n_anomalies)  # Pressure spikes
        vibration_anomaly = np.random.exponential(20, n_anomalies)  # High vibration
        power_anomaly = np.random.uniform(200, 500, n_anomalies)  # Power surges

        # Combine data
        temps = np.concatenate([temp_normal, temp_anomaly])
        pressures = np.concatenate([pressure_normal, pressure_anomaly])
        vibrations = np.concatenate([vibration_normal, vibration_anomaly])
        powers = np.concatenate([power_normal, power_anomaly])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle
        indices = np.random.permutation(len(temps))

        df = pd.DataFrame(
            {
                "temperature_celsius": temps[indices],
                "pressure_bar": pressures[indices],
                "vibration_mm_per_sec": vibrations[indices],
                "power_consumption_kw": powers[indices],
            }
        )

        dataset = Dataset(
            name="Industrial IoT Sensors",
            data=df,
            description="Manufacturing equipment sensor readings",
        )

        return dataset, labels[indices]

    def monitor_equipment(self, dataset, true_labels):
        """Run equipment monitoring analysis."""
        print(f"\n🏭 {self.name} Analysis")
        print("-" * 50)

        # Use fastest algorithm for real-time monitoring
        detector = PyODAdapter(
            algorithm_name="ECOD",
            contamination_rate=ContaminationRate(self.contamination_rate),
        )

        detector.fit(dataset)
        result = detector.detect(dataset)

        accuracy = np.mean(result.labels == true_labels)

        print("⚙️ Equipment Health Monitoring:")
        print(f"   Total readings: {len(dataset.data)}")
        print(f"   Anomalies detected: {len(result.anomalies)}")
        print(f"   Detection accuracy: {accuracy:.1%}")
        print(
            f"   Processing speed: {len(dataset.data)/result.execution_time_ms*1000:.0f} readings/sec"
        )

        # Identify critical alerts
        scores = np.array([score.value for score in result.scores])
        critical_indices = np.where((result.labels == 1) & (scores > 0.8))[0]

        print("\n🚨 Critical Equipment Alerts:")
        for i, idx in enumerate(critical_indices[:3], 1):
            sample = dataset.data.iloc[idx]
            score = scores[idx]
            print(f"   {i}. Reading {idx}: Score={score:.3f}")
            print(
                f"      Temp: {sample['temperature_celsius']:.1f}°C, "
                f"Pressure: {sample['pressure_bar']:.1f} bar, "
                f"Vibration: {sample['vibration_mm_per_sec']:.1f} mm/s"
            )

        return result


class CybersecurityThreatDetector:
    """Cybersecurity threat detection template."""

    def __init__(self):
        self.name = "Cybersecurity Threat Detection"
        self.recommended_algorithms = ["IsolationForest", "LOF", "ECOD"]
        self.contamination_rate = 0.02  # 2% attack rate

    def create_sample_data(self, n_samples=8000):
        """Create sample network traffic data."""
        np.random.seed(42)

        print(f"🛡️ Generating {n_samples} network traffic logs...")

        # Normal traffic (98%)
        n_normal = int(n_samples * 0.98)

        # Packet size (bytes) - mostly small packets
        packet_size_normal = np.random.lognormal(8, 1, n_normal)  # ~3KB average
        packet_size_normal = np.clip(packet_size_normal, 64, 65536)

        # Connection duration (seconds)
        duration_normal = np.random.exponential(30, n_normal)  # 30s average

        # Bytes transferred
        bytes_normal = packet_size_normal * np.random.poisson(10, n_normal)

        # Port diversity (unique ports accessed)
        ports_normal = np.random.poisson(3, n_normal)  # Few ports normal

        # Attack traffic (2%)
        n_attacks = n_samples - n_normal

        # Attack patterns: DDoS, data exfiltration, port scanning
        attack_type = np.random.choice([0, 1, 2], n_attacks)
        packet_size_attack = np.zeros(n_attacks)
        for i, atype in enumerate(attack_type):
            if atype == 0:
                packet_size_attack[i] = np.random.uniform(64, 128)  # Small DDoS packets
            elif atype == 1:
                packet_size_attack[i] = np.random.uniform(
                    32768, 65536
                )  # Large data transfers
            else:
                packet_size_attack[i] = np.random.uniform(40, 80)  # SYN flood

        duration_attack = np.random.choice(
            [
                np.random.uniform(0.1, 1, n_attacks // 2),  # Very short (scanning)
                np.random.uniform(
                    300, 3600, n_attacks // 2
                ),  # Very long (exfiltration)
            ],
            n_attacks,
        )

        bytes_attack = np.random.uniform(1e6, 1e9, n_attacks)  # Large transfers
        ports_attack = np.random.poisson(50, n_attacks)  # Many ports (scanning)

        # Combine data
        packet_sizes = np.concatenate([packet_size_normal, packet_size_attack])
        durations = np.concatenate([duration_normal, duration_attack])
        bytes_transferred = np.concatenate([bytes_normal, bytes_attack])
        port_counts = np.concatenate([ports_normal, ports_attack])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_attacks)])

        # Shuffle
        indices = np.random.permutation(len(packet_sizes))

        df = pd.DataFrame(
            {
                "avg_packet_size_bytes": packet_sizes[indices],
                "connection_duration_sec": durations[indices],
                "total_bytes_transferred": bytes_transferred[indices],
                "unique_ports_accessed": port_counts[indices],
            }
        )

        dataset = Dataset(
            name="Network Traffic Logs",
            data=df,
            description="Network connections with attack indicators",
        )

        return dataset, labels[indices]

    def detect_threats(self, dataset, true_labels):
        """Run cybersecurity threat detection."""
        print(f"\n🛡️ {self.name} Analysis")
        print("-" * 50)

        # Use IsolationForest for network anomalies
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(self.contamination_rate),
            random_state=42,
        )

        detector.fit(dataset)
        result = detector.detect(dataset)

        accuracy = np.mean(result.labels == true_labels)
        precision = np.sum((result.labels == 1) & (true_labels == 1)) / max(
            np.sum(result.labels == 1), 1
        )

        print("🔒 Network Security Analysis:")
        print(f"   Total connections: {len(dataset.data)}")
        print(f"   Threats detected: {len(result.anomalies)}")
        print(f"   Detection accuracy: {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(
            f"   Analysis speed: {len(dataset.data)/result.execution_time_ms*1000:.0f} connections/sec"
        )

        # Classify threat types
        scores = np.array([score.value for score in result.scores])
        threat_indices = np.where(result.labels == 1)[0]

        print("\n🚨 Top Security Threats:")
        for i, idx in enumerate(
            threat_indices[np.argsort(scores[threat_indices])[-3:]][::-1], 1
        ):
            sample = dataset.data.iloc[idx]
            score = scores[idx]

            # Simple threat classification
            if sample["unique_ports_accessed"] > 20:
                threat_type = "Port Scanning"
            elif sample["total_bytes_transferred"] > 1e8:
                threat_type = "Data Exfiltration"
            elif sample["avg_packet_size_bytes"] < 100:
                threat_type = "DDoS Attack"
            else:
                threat_type = "Unknown Threat"

            print(f"   {i}. {threat_type}: Score={score:.3f}")
            print(
                f"      Duration: {sample['connection_duration_sec']:.1f}s, "
                f"Bytes: {sample['total_bytes_transferred']/1e6:.1f}MB, "
                f"Ports: {sample['unique_ports_accessed']}"
            )

        return result


def run_industry_templates():
    """Run all industry-specific templates."""
    print("🏭 Pynomaly Industry Templates")
    print("=" * 60)
    print("Demonstrating domain-specific anomaly detection solutions")

    # Financial Fraud Detection
    fraud_detector = FinancialFraudDetector()
    fraud_dataset, fraud_labels = fraud_detector.create_sample_data()
    fraud_results = fraud_detector.detect_fraud(fraud_dataset, fraud_labels)

    # Industrial IoT Monitoring
    iot_monitor = IndustrialIoTMonitor()
    iot_dataset, iot_labels = iot_monitor.create_sample_data()
    iot_results = iot_monitor.monitor_equipment(iot_dataset, iot_labels)

    # Cybersecurity Threat Detection
    security_detector = CybersecurityThreatDetector()
    security_dataset, security_labels = security_detector.create_sample_data()
    security_results = security_detector.detect_threats(
        security_dataset, security_labels
    )

    # Summary
    print("\n🎯 INDUSTRY TEMPLATES SUMMARY")
    print("=" * 60)

    print("✅ Financial Fraud Detection:")
    best_fraud = max(fraud_results, key=lambda x: x["precision"])
    print(f"   Best Algorithm: {best_fraud['algorithm']}")
    print(f"   Precision: {best_fraud['precision']:.1%} (Critical for fraud)")
    print(f"   Processing: {best_fraud['execution_time']:.3f}s for 10K transactions")

    print("\n✅ Industrial IoT Monitoring:")
    print("   Algorithm: ECOD (Optimized for speed)")
    print("   Real-time capability: ✓")
    print("   Equipment failure detection: ✓")

    print("\n✅ Cybersecurity Threat Detection:")
    print("   Algorithm: IsolationForest")
    print("   Multi-threat detection: ✓")
    print("   Network-scale processing: ✓")

    print("\n🚀 Key Benefits:")
    print("   📊 Domain-optimized algorithms")
    print("   ⚡ Real-time processing capabilities")
    print("   🎯 Industry-specific metrics and alerts")
    print("   🔧 Ready-to-deploy templates")

    print("\n💡 Each template can be customized for specific:")
    print("   • Data formats and preprocessing")
    print("   • Algorithm parameters and thresholds")
    print("   • Alert systems and integrations")
    print("   • Compliance and regulatory requirements")


def main():
    """Main function."""
    try:
        run_industry_templates()
        return 0
    except Exception as e:
        print(f"❌ Industry templates failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
