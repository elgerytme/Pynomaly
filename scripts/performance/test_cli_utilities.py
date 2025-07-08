#!/usr/bin/env python3
"""
Simple test to verify CLI utilities work correctly.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def create_sample_baseline():
    """Create a sample baseline.json for testing."""
    return {
        "performance_metrics": {
            "IsolationForest": [
                {
                    "algorithm_name": "IsolationForest",
                    "execution_time_seconds": 2.5,
                    "peak_memory_mb": 150.0,
                    "training_throughput": 4000.0,
                    "accuracy_score": 0.85,
                    "success": True,
                }
            ],
            "LocalOutlierFactor": [
                {
                    "algorithm_name": "LocalOutlierFactor",
                    "execution_time_seconds": 1.8,
                    "peak_memory_mb": 120.0,
                    "training_throughput": 5000.0,
                    "accuracy_score": 0.82,
                    "success": True,
                }
            ],
        }
    }


def create_sample_current():
    """Create a sample current.json for testing."""
    return {
        "performance_metrics": {
            "IsolationForest": [
                {
                    "algorithm_name": "IsolationForest",
                    "execution_time_seconds": 3.2,  # 28% slower - major regression
                    "peak_memory_mb": 180.0,  # 20% more memory - major regression
                    "training_throughput": 3200.0,  # 20% slower throughput - major regression
                    "accuracy_score": 0.83,  # 2.4% lower accuracy - minor regression
                    "success": True,
                }
            ],
            "LocalOutlierFactor": [
                {
                    "algorithm_name": "LocalOutlierFactor",
                    "execution_time_seconds": 1.6,  # 11% faster - minor improvement
                    "peak_memory_mb": 115.0,  # 4% less memory - minor improvement
                    "training_throughput": 5200.0,  # 4% faster throughput - minor improvement
                    "accuracy_score": 0.84,  # 2.4% better accuracy - minor improvement
                    "success": True,
                }
            ],
        }
    }


def test_check_regressions():
    """Test the check_regressions.py script."""
    print("Testing check_regressions.py...")

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample data files
        baseline_file = temp_path / "baseline.json"
        current_file = temp_path / "current.json"
        config_file = Path(__file__).parent / "performance_config.yml"

        with open(baseline_file, "w") as f:
            json.dump(create_sample_baseline(), f, indent=2)

        with open(current_file, "w") as f:
            json.dump(create_sample_current(), f, indent=2)

        # Run check_regressions.py
        script_path = Path(__file__).parent / "check_regressions.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_file),
            "--current",
            str(current_file),
            "--config",
            str(config_file),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            print(f"Exit code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")

            # Check if markdown output contains expected content
            if "Performance Regression Summary" in result.stdout:
                print("✓ Markdown summary generated successfully")
            else:
                print("✗ Markdown summary not found in output")

            # Should exit with 1 if critical regressions found
            if result.returncode == 1:
                print("✓ Script correctly exits with code 1 for regressions")
            else:
                print("✗ Script did not exit with expected code")

        except subprocess.TimeoutExpired:
            print("✗ Script timed out")
        except Exception as e:
            print(f"✗ Error running script: {e}")


def test_run_benchmarks():
    """Test the run_benchmarks.py script with dry-run mode."""
    print("\nTesting run_benchmarks.py...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_file = temp_path / "test_results.json"

        script_path = Path(__file__).parent / "run_benchmarks.py"
        config_file = Path(__file__).parent / "performance_config.yml"

        # Test help option first
        cmd = [sys.executable, str(script_path), "--help"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and "performance benchmarks" in result.stdout:
                print("✓ Help option works correctly")
            else:
                print("✗ Help option failed")
                print(f"Exit code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

        except Exception as e:
            print(f"✗ Error testing help option: {e}")


def main():
    """Run all tests."""
    print("Testing Performance CLI Utilities")
    print("=" * 40)

    test_check_regressions()
    test_run_benchmarks()

    print("\nTests completed.")


if __name__ == "__main__":
    main()
