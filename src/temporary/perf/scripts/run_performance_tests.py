#!/usr/bin/env python3
"""
Performance test runner with containerization support.

This script runs performance tests in a controlled environment with
resource limits and statistical baseline comparisons.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_performance_tests(use_container=False):
    """Run performance tests with resource limits."""

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent / "src")
    env["PERF_TESTING"] = "true"

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define test command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(Path(__file__).parent.parent / "tests" / "test_performance_simple.py"),
        "-v",
        "--benchmark-only",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=3",
        "--benchmark-min-rounds=5",
        "--benchmark-max-time=60",
        "--benchmark-sort=mean",
        f'--benchmark-json={results_dir / "benchmark_results.json"}',
        "--benchmark-save-data",
        "--tb=short",
    ]

    if use_container:
        print("Running performance tests in containerized environment...")
        # Use docker-compose to run tests
        container_cmd = [
            "docker-compose",
            "-f",
            str(Path(__file__).parent.parent / "perf" / "docker-compose.perf.yml"),
            "up",
            "--build",
        ]
        result = subprocess.run(container_cmd, capture_output=True, text=True)
    else:
        print("Running performance tests locally...")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

    return result.returncode == 0


def generate_performance_report():
    """Generate performance test report."""
    results_file = Path(__file__).parent.parent / "results" / "benchmark_results.json"

    if not results_file.exists():
        print("No benchmark results found.")
        return

    with open(results_file) as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("PERFORMANCE TEST REPORT")
    print("=" * 60)

    for benchmark in data.get("benchmarks", []):
        name = benchmark["name"]
        stats = benchmark["stats"]

        print(f"\nTest: {name}")
        print(f"  Median: {stats['median']*1000:.2f}ms")
        print(f"  Mean: {stats['mean']*1000:.2f}ms")
        print(f"  Std Dev: {stats['stddev']*1000:.2f}ms")
        print(f"  Min: {stats['min']*1000:.2f}ms")
        print(f"  Max: {stats['max']*1000:.2f}ms")
        print(f"  Rounds: {stats['rounds']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run performance tests")
    parser.add_argument(
        "--container",
        action="store_true",
        help="Run tests in containerized environment",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report from existing results",
    )

    args = parser.parse_args()

    if args.report:
        generate_performance_report()
    else:
        success = run_performance_tests(use_container=args.container)
        if success:
            generate_performance_report()
        else:
            print("Performance tests failed!")
            sys.exit(1)
