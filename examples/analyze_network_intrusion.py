#!/usr/bin/env python3
"""
Network Intrusion Dataset Analysis Example

Demonstrates how to analyze network traffic data for intrusion detection
using Pynomaly's algorithms and techniques.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add Pynomaly to path (adjust if needed)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_network_intrusion_data():
    """Load the network intrusion dataset"""
    data_path = (
        Path(__file__).parent
        / "sample_datasets"
        / "synthetic"
        / "network_intrusion.csv"
    )

    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        print("Please run scripts/generate_comprehensive_datasets.py first")
        return None

    df = pd.read_csv(data_path)
    print(
        f"✅ Loaded network intrusion dataset: {len(df)} samples, {len(df.columns) - 1} features"
    )
    return df


def network_traffic_analysis(df):
    """Analyze network traffic patterns"""
    print("\n" + "=" * 60)
    print("NETWORK TRAFFIC ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Intrusion rate: {df['is_anomaly'].mean():.2%}")
    print(f"Total intrusions detected: {df['is_anomaly'].sum()}")

    # Protocol analysis
    print("\nProtocol Distribution:")
    protocol_dist = df.groupby(["protocol", "is_anomaly"]).size().unstack(fill_value=0)
    protocol_rates = protocol_dist.div(protocol_dist.sum(axis=1), axis=0)

    for protocol in protocol_dist.index:
        normal_count = (
            protocol_dist.loc[protocol, 0] if 0 in protocol_dist.columns else 0
        )
        attack_count = (
            protocol_dist.loc[protocol, 1] if 1 in protocol_dist.columns else 0
        )
        total = normal_count + attack_count
        attack_rate = attack_count / total if total > 0 else 0
        print(f"  {protocol}: {total} connections, {attack_rate:.1%} attack rate")

    # Traffic volume analysis
    print("\nTraffic Volume Analysis:")
    normal_traffic = df[df["is_anomaly"] == 0]
    attack_traffic = df[df["is_anomaly"] == 1]

    volume_metrics = [
        "packet_count",
        "bytes_transferred",
        "packets_per_second",
        "bytes_per_packet",
    ]

    for metric in volume_metrics:
        normal_mean = normal_traffic[metric].mean()
        normal_std = normal_traffic[metric].std()
        attack_mean = attack_traffic[metric].mean()
        attack_std = attack_traffic[metric].std()

        print(f"  {metric}:")
        print(f"    Normal: {normal_mean:.2f} ± {normal_std:.2f}")
        print(f"    Attack: {attack_mean:.2f} ± {attack_std:.2f}")
        print(f"    Ratio: {attack_mean / normal_mean:.2f}x")

    # Port analysis
    print("\nPort Analysis:")

    # Standard ports (well-known)
    standard_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
    normal_std_ports = normal_traffic[
        normal_traffic["destination_port"].isin(standard_ports)
    ]["destination_port"].value_counts()
    attack_std_ports = attack_traffic[
        attack_traffic["destination_port"].isin(standard_ports)
    ]["destination_port"].value_counts()

    print(f"  Top normal traffic ports: {list(normal_std_ports.head().index)}")
    print(f"  Top attack traffic ports: {list(attack_std_ports.head().index)}")

    # High ports (>1024) - often used in attacks
    high_port_normal_rate = (normal_traffic["destination_port"] > 1024).mean()
    high_port_attack_rate = (attack_traffic["destination_port"] > 1024).mean()
    print("  High port usage (>1024):")
    print(f"    Normal traffic: {high_port_normal_rate:.1%}")
    print(f"    Attack traffic: {high_port_attack_rate:.1%}")

    return {
        "protocol_dist": protocol_dist,
        "normal_traffic": normal_traffic,
        "attack_traffic": attack_traffic,
        "volume_metrics": volume_metrics,
    }


def intrusion_pattern_detection(df):
    """Detect and analyze intrusion patterns"""
    print("\n" + "=" * 60)
    print("INTRUSION PATTERN DETECTION")
    print("=" * 60)

    # DDoS pattern detection (high packet rate, short duration)
    high_packet_rate = df["packets_per_second"] > df["packets_per_second"].quantile(
        0.95
    )
    short_duration = df["duration_seconds"] < df["duration_seconds"].quantile(0.25)

    ddos_pattern = high_packet_rate & short_duration
    ddos_attack_rate = df[ddos_pattern]["is_anomaly"].mean()

    print("🔍 DDoS Pattern Detection:")
    print("  Pattern criteria: High packet rate (>95th %) + Short duration (<25th %)")
    print(f"  Connections matching pattern: {ddos_pattern.sum()}")
    print(f"  Attack rate in pattern: {ddos_attack_rate:.1%}")

    # Port scanning pattern (many different ports, small packets)
    port_variety = df.groupby(df.index // 100)[
        "destination_port"
    ].nunique()  # Port variety in chunks
    small_packets = df["bytes_per_packet"] < df["bytes_per_packet"].quantile(0.25)

    print("\n🔍 Port Scanning Indicators:")
    print(f"  Small packet size (<25th %): {small_packets.sum()} connections")
    print(
        f"  Attack rate in small packets: {df[small_packets]['is_anomaly'].mean():.1%}"
    )

    # Data exfiltration pattern (large bytes, long duration)
    large_transfer = df["bytes_transferred"] > df["bytes_transferred"].quantile(0.95)
    long_duration = df["duration_seconds"] > df["duration_seconds"].quantile(0.95)

    exfiltration_pattern = large_transfer & long_duration
    exfiltration_attack_rate = df[exfiltration_pattern]["is_anomaly"].mean()

    print("\n🔍 Data Exfiltration Pattern:")
    print("  Pattern criteria: Large transfer (>95th %) + Long duration (>95th %)")
    print(f"  Connections matching pattern: {exfiltration_pattern.sum()}")
    print(f"  Attack rate in pattern: {exfiltration_attack_rate:.1%}")

    # Unusual protocol usage
    unusual_protocols = df["protocol"].value_counts()
    minority_protocols = unusual_protocols[
        unusual_protocols < unusual_protocols.quantile(0.1)
    ].index

    if len(minority_protocols) > 0:
        minority_usage = df["protocol"].isin(minority_protocols)
        minority_attack_rate = df[minority_usage]["is_anomaly"].mean()

        print("\n🔍 Unusual Protocol Usage:")
        print(f"  Minority protocols: {list(minority_protocols)}")
        print(f"  Attack rate in minority protocols: {minority_attack_rate:.1%}")

    return {
        "ddos_pattern": ddos_pattern,
        "small_packets": small_packets,
        "exfiltration_pattern": exfiltration_pattern,
        "ddos_attack_rate": ddos_attack_rate,
        "exfiltration_attack_rate": exfiltration_attack_rate,
    }


def feature_engineering_network(df):
    """Engineer network-specific features"""
    print("\n" + "=" * 60)
    print("NETWORK FEATURE ENGINEERING")
    print("=" * 60)

    df_enhanced = df.copy()

    # Traffic intensity features
    df_enhanced["traffic_intensity"] = (
        df_enhanced["packet_count"] * df_enhanced["packets_per_second"]
    )
    df_enhanced["bandwidth_usage"] = df_enhanced["bytes_transferred"] / (
        df_enhanced["duration_seconds"] + 0.001
    )

    # Protocol risk features
    high_risk_protocols = ["ICMP"]  # ICMP often used in attacks
    df_enhanced["is_high_risk_protocol"] = (
        df_enhanced["protocol"].isin(high_risk_protocols).astype(int)
    )

    # Port classification features
    df_enhanced["is_well_known_port"] = (
        df_enhanced["destination_port"] <= 1023
    ).astype(int)
    df_enhanced["is_registered_port"] = (
        (df_enhanced["destination_port"] >= 1024)
        & (df_enhanced["destination_port"] <= 49151)
    ).astype(int)
    df_enhanced["is_dynamic_port"] = (df_enhanced["destination_port"] > 49151).astype(
        int
    )

    # Traffic pattern features
    df_enhanced["is_high_packet_rate"] = (
        df_enhanced["packets_per_second"]
        > df_enhanced["packets_per_second"].quantile(0.9)
    ).astype(int)
    df_enhanced["is_large_transfer"] = (
        df_enhanced["bytes_transferred"]
        > df_enhanced["bytes_transferred"].quantile(0.9)
    ).astype(int)
    df_enhanced["is_short_connection"] = (df_enhanced["duration_seconds"] < 1).astype(
        int
    )
    df_enhanced["is_long_connection"] = (
        df_enhanced["duration_seconds"] > df_enhanced["duration_seconds"].quantile(0.9)
    ).astype(int)

    # Efficiency features
    df_enhanced["packet_efficiency"] = df_enhanced["bytes_transferred"] / (
        df_enhanced["packet_count"] + 1
    )
    df_enhanced["time_efficiency"] = df_enhanced["packet_count"] / (
        df_enhanced["duration_seconds"] + 0.001
    )

    # Anomaly score features (simplified scoring)
    df_enhanced["volume_score"] = (
        (df_enhanced["packet_count"] - df_enhanced["packet_count"].mean())
        / df_enhanced["packet_count"].std()
    ).abs()

    df_enhanced["timing_score"] = (
        (df_enhanced["duration_seconds"] - df_enhanced["duration_seconds"].mean())
        / df_enhanced["duration_seconds"].std()
    ).abs()

    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"Created {len(new_features)} new features:")
    for feature in new_features:
        print(f"  • {feature}")

    # Feature importance analysis
    print("\nFeature correlation with intrusions:")
    correlation_features = new_features[:5]  # Top 5 new features
    for feature in correlation_features:
        corr = df_enhanced[feature].corr(df_enhanced["is_anomaly"])
        print(f"  {feature}: {corr:.3f}")

    return df_enhanced


def algorithm_recommendations_network(df):
    """Provide algorithm recommendations for network intrusion detection"""
    print("\n" + "=" * 60)
    print("ALGORITHM RECOMMENDATIONS")
    print("=" * 60)

    # Data characteristics analysis
    n_features = len(df.columns) - 1  # Excluding label
    n_samples = len(df)
    contamination_rate = df["is_anomaly"].mean()

    print("📊 Dataset Characteristics:")
    print(f"  • Samples: {n_samples:,}")
    print(f"  • Features: {n_features}")
    print(f"  • Contamination rate: {contamination_rate:.1%}")
    print("  • Feature types: Mixed (numerical + categorical)")
    print("  • Data complexity: Medium-High")

    # Algorithm suitability analysis
    algorithms = [
        {
            "name": "IsolationForest",
            "suitability": 0.90,
            "pros": [
                "Excellent for network data",
                "Handles mixed features well",
                "Fast training",
            ],
            "cons": ["May miss subtle patterns", "Parameter tuning needed"],
            "use_case": "General intrusion detection, DDoS detection",
        },
        {
            "name": "LocalOutlierFactor",
            "suitability": 0.85,
            "pros": [
                "Good for local anomalies",
                "Effective for port scanning",
                "No assumptions about data",
            ],
            "cons": ["Sensitive to parameters", "Slower on large datasets"],
            "use_case": "Port scanning, local attack patterns",
        },
        {
            "name": "EllipticEnvelope",
            "suitability": 0.70,
            "pros": ["Good for Gaussian data", "Robust to outliers", "Fast prediction"],
            "cons": [
                "Assumes elliptical distribution",
                "May not fit network data well",
            ],
            "use_case": "Clean network environments, baseline detection",
        },
        {
            "name": "OneClassSVM",
            "suitability": 0.75,
            "pros": [
                "Non-linear boundaries",
                "Kernel flexibility",
                "Good generalization",
            ],
            "cons": [
                "Slow on large datasets",
                "Memory intensive",
                "Parameter sensitive",
            ],
            "use_case": "Complex attack patterns, non-linear boundaries",
        },
    ]

    # Sort by suitability
    algorithms.sort(key=lambda x: x["suitability"], reverse=True)

    print("\n🧠 Algorithm Recommendations (ranked by suitability):")

    for i, algo in enumerate(algorithms, 1):
        print(f"\n{i}. {algo['name']} (Suitability: {algo['suitability']:.0%})")
        print(f"   Use case: {algo['use_case']}")
        print(f"   Pros: {', '.join(algo['pros'])}")
        print(f"   Cons: {', '.join(algo['cons'])}")

    print("\n💡 Implementation Strategy:")
    print("  1. Start with IsolationForest for general detection")
    print("  2. Use LOF for detailed local pattern analysis")
    print("  3. Consider ensemble methods for comprehensive coverage")
    print("  4. Implement real-time streaming detection")
    print("  5. Use different algorithms for different attack types")

    return algorithms


def real_time_detection_strategy(df):
    """Outline real-time detection strategy"""
    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION STRATEGY")
    print("=" * 60)

    # Analysis of timing requirements
    connection_durations = df["duration_seconds"]
    quick_connections = (connection_durations < 1).sum()
    medium_connections = (
        (connection_durations >= 1) & (connection_durations < 10)
    ).sum()
    long_connections = (connection_durations >= 10).sum()

    print("⏱️  Connection Duration Analysis:")
    print(f"  • Quick (<1s): {quick_connections} ({quick_connections / len(df):.1%})")
    print(
        f"  • Medium (1-10s): {medium_connections} ({medium_connections / len(df):.1%})"
    )
    print(f"  • Long (>10s): {long_connections} ({long_connections / len(df):.1%})")

    # Detection strategy by connection type
    print("\n🎯 Detection Strategy by Connection Type:")

    print(
        f"\n1. Quick Connections (<1s) - {quick_connections / len(df):.1%} of traffic:"
    )
    print("   • Challenge: Limited data for analysis")
    print("   • Approach: Focus on packet patterns and port scanning")
    print("   • Algorithm: IsolationForest with minimal features")
    print("   • Threshold: Lower sensitivity to avoid false positives")

    print(
        f"\n2. Medium Connections (1-10s) - {medium_connections / len(df):.1%} of traffic:"
    )
    print("   • Challenge: Balance speed vs accuracy")
    print("   • Approach: Multi-stage detection (quick + detailed)")
    print("   • Algorithm: IsolationForest + LOF for suspicious connections")
    print("   • Threshold: Adaptive based on traffic patterns")

    print(
        f"\n3. Long Connections (>10s) - {long_connections / len(df):.1%} of traffic:"
    )
    print("   • Challenge: Data exfiltration and APT detection")
    print("   • Approach: Comprehensive analysis with all features")
    print("   • Algorithm: Ensemble methods with multiple algorithms")
    print("   • Threshold: High sensitivity for persistent threats")

    # Implementation architecture
    print("\n🏗️  Implementation Architecture:")
    print("  1. Stream Processing:")
    print("     • Kafka/Pulsar for high-throughput ingestion")
    print("     • Sliding window aggregation")
    print("     • Feature extraction in real-time")

    print("\n  2. Multi-tier Detection:")
    print("     • Tier 1: Fast rule-based screening")
    print("     • Tier 2: ML-based anomaly scoring")
    print("     • Tier 3: Deep analysis for high-risk connections")

    print("\n  3. Response System:")
    print("     • Automated blocking for high-confidence threats")
    print("     • Alert generation for suspicious activity")
    print("     • Forensic data collection for investigation")


def create_network_visualizations(df, analysis_results):
    """Create visualizations for network analysis"""
    print("\n" + "=" * 60)
    print("CREATING NETWORK VISUALIZATIONS")
    print("=" * 60)

    try:
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Network Intrusion Dataset Analysis", fontsize=16)

        # 1. Protocol distribution
        protocol_data = analysis_results["protocol_dist"]
        protocol_data.plot(kind="bar", ax=axes[0, 0], stacked=True)
        axes[0, 0].set_title("Protocol Distribution (Normal vs Attack)")
        axes[0, 0].set_ylabel("Connection Count")
        axes[0, 0].legend(["Normal", "Attack"])
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Packet count distribution
        normal_packets = analysis_results["normal_traffic"]["packet_count"]
        attack_packets = analysis_results["attack_traffic"]["packet_count"]

        axes[0, 1].hist(
            normal_packets,
            bins=50,
            alpha=0.7,
            label="Normal",
            density=True,
            range=(0, np.percentile(normal_packets, 95)),
        )
        axes[0, 1].hist(
            attack_packets,
            bins=50,
            alpha=0.7,
            label="Attack",
            density=True,
            range=(0, np.percentile(attack_packets, 95)),
        )
        axes[0, 1].set_xlabel("Packet Count")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Packet Count Distribution")
        axes[0, 1].legend()

        # 3. Bytes per packet
        normal_bpp = analysis_results["normal_traffic"]["bytes_per_packet"]
        attack_bpp = analysis_results["attack_traffic"]["bytes_per_packet"]

        axes[1, 0].hist(
            normal_bpp,
            bins=50,
            alpha=0.7,
            label="Normal",
            density=True,
            range=(0, np.percentile(normal_bpp, 95)),
        )
        axes[1, 0].hist(
            attack_bpp,
            bins=50,
            alpha=0.7,
            label="Attack",
            density=True,
            range=(0, np.percentile(attack_bpp, 95)),
        )
        axes[1, 0].set_xlabel("Bytes per Packet")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Bytes per Packet Distribution")
        axes[1, 0].legend()

        # 4. Connection duration vs packets per second scatter
        sample_size = min(1000, len(df))  # Sample for performance
        sample_df = df.sample(n=sample_size)

        normal_sample = sample_df[sample_df["is_anomaly"] == 0]
        attack_sample = sample_df[sample_df["is_anomaly"] == 1]

        axes[1, 1].scatter(
            normal_sample["duration_seconds"],
            normal_sample["packets_per_second"],
            alpha=0.6,
            label="Normal",
            s=20,
        )
        axes[1, 1].scatter(
            attack_sample["duration_seconds"],
            attack_sample["packets_per_second"],
            alpha=0.8,
            label="Attack",
            s=20,
            color="red",
        )
        axes[1, 1].set_xlabel("Duration (seconds)")
        axes[1, 1].set_ylabel("Packets per Second")
        axes[1, 1].set_title("Connection Patterns (Duration vs Packet Rate)")
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, np.percentile(sample_df["duration_seconds"], 95))
        axes[1, 1].set_ylim(0, np.percentile(sample_df["packets_per_second"], 95))

        plt.tight_layout()

        # Save plot
        output_path = Path(__file__).parent / "network_intrusion_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"📊 Visualizations saved to: {output_path}")

        # Don't show plot in automated runs
        if len(sys.argv) == 1:  # Only show if run directly
            plt.show()

    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")


def main():
    """Main analysis workflow"""
    print("🔒 NETWORK INTRUSION DATASET ANALYSIS")
    print("=" * 60)
    print("This example demonstrates network intrusion detection using Pynomaly")
    print("with network traffic data containing various attack patterns.")

    # Load data
    df = load_network_intrusion_data()
    if df is None:
        return

    # Network traffic analysis
    analysis_results = network_traffic_analysis(df)

    # Intrusion pattern detection
    pattern_results = intrusion_pattern_detection(df)

    # Feature engineering
    df_enhanced = feature_engineering_network(df)

    # Algorithm recommendations
    algorithms = algorithm_recommendations_network(df_enhanced)

    # Real-time detection strategy
    real_time_detection_strategy(df)

    # Create visualizations
    create_network_visualizations(df, analysis_results)

    print("\n✅ Analysis complete!")
    print("📋 Key takeaways:")
    print(f"   • {df['is_anomaly'].mean():.1%} intrusion rate in dataset")
    print("   • IsolationForest recommended as primary algorithm")
    print("   • DDoS patterns: High packet rate + short duration")
    print("   • Port scanning: Small packets + port variety")
    print("   • Multi-tier real-time detection recommended")


if __name__ == "__main__":
    main()
