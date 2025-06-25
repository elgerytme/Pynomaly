#!/usr/bin/env python3
"""
Pynomaly Quickstart Example
Demonstrates basic usage of Pynomaly CLI for anomaly detection
"""

import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a CLI command and display results."""
    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}")
    print("-" * 40)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """Run the quickstart example."""
    print("🚀 Pynomaly Quickstart Example")
    print("This script demonstrates basic Pynomaly CLI usage")

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples" / "datasets"

    # Change to project directory for CLI commands
    import os

    os.chdir(project_root)

    # Example 1: System Check
    run_command("python3 pynomaly_cli.py perf-stats", "System Performance Check")

    # Example 2: View Available Algorithms
    run_command(
        "python3 pynomaly_cli.py detector-list",
        "Available Anomaly Detection Algorithms",
    )

    # Example 3: Examine Sample Dataset
    dataset_path = examples_dir / "simple_anomalies.csv"
    run_command(
        f"python3 pynomaly_cli.py dataset-info {dataset_path}",
        "Dataset Information Analysis",
    )

    # Example 4: Data Quality Validation
    run_command(
        f"python3 pynomaly_cli.py validate {dataset_path}", "Data Quality Validation"
    )

    # Example 5: Basic Anomaly Detection
    run_command(
        f"python3 pynomaly_cli.py detect {dataset_path}",
        "Basic Anomaly Detection (Default Settings)",
    )

    # Example 6: Anomaly Detection with Specific Algorithm
    run_command(
        f"python3 pynomaly_cli.py detect {dataset_path} LocalOutlierFactor 0.15",
        "Anomaly Detection with LocalOutlierFactor",
    )

    # Example 7: Algorithm Performance Benchmark
    run_command(
        f"python3 pynomaly_cli.py benchmark {dataset_path}",
        "Algorithm Performance Benchmark",
    )

    # Example 8: Time Series Dataset Analysis
    time_series_path = examples_dir / "time_series_anomalies.csv"
    run_command(
        f"python3 pynomaly_cli.py detect {time_series_path} IsolationForest 0.1",
        "Time Series Anomaly Detection",
    )

    # Example 9: Credit Card Fraud Detection
    credit_path = examples_dir / "credit_card_sample.csv"
    run_command(
        f"python3 pynomaly_cli.py detect {credit_path} OneClassSVM 0.1",
        "Credit Card Fraud Detection",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("🎉 Quickstart Example Complete!")
    print(f"{'=' * 60}")
    print("Key takeaways:")
    print("• Pynomaly provides a simple CLI interface for anomaly detection")
    print("• Multiple algorithms are available for different use cases")
    print("• Data validation helps ensure quality results")
    print("• Performance monitoring guides system optimization")
    print("• Benchmark comparisons help select the best algorithm")
    print("\nNext steps:")
    print("• Try the examples with your own datasets")
    print("• Explore the API server: python3 pynomaly_cli.py server-start")
    print("• Read the full usage guide in examples/USAGE_GUIDE.md")


if __name__ == "__main__":
    main()
