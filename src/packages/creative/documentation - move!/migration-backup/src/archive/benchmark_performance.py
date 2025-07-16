#!/usr/bin/env python3
"""
Performance benchmarking for Pynomaly detection algorithms.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter


def generate_test_data(n_samples=1000, n_features=10, contamination=0.1):
    """Generate synthetic test data with outliers."""
    np.random.seed(42)

    # Generate normal data
    normal_samples = int(n_samples * (1 - contamination))
    normal_data = np.random.normal(0, 1, (normal_samples, n_features))

    # Generate outliers
    outlier_samples = n_samples - normal_samples
    outlier_data = np.random.uniform(-5, 5, (outlier_samples, n_features))

    # Combine data
    data = np.vstack([normal_data, outlier_data])

    # Create DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    return Dataset(name="Benchmark Data", data=df)


def benchmark_sklearn_algorithms():
    """Benchmark sklearn-based algorithms."""
    print("🔍 Benchmarking Sklearn Algorithms")
    print("=" * 50)

    # Test configurations
    data_sizes = [100, 500, 1000, 5000]
    algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

    results = []

    for n_samples in data_sizes:
        dataset = generate_test_data(n_samples=n_samples, n_features=10)

        for algorithm in algorithms:
            print(f"Testing {algorithm} with {n_samples} samples...")

            try:
                # Create adapter
                if algorithm == "LocalOutlierFactor":
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm,
                        name=f"Benchmark {algorithm}",
                        contamination_rate=ContaminationRate(0.1),
                    )
                elif algorithm == "OneClassSVM":
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm,
                        name=f"Benchmark {algorithm}",
                        contamination_rate=ContaminationRate(0.1),
                    )
                else:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm,
                        name=f"Benchmark {algorithm}",
                        contamination_rate=ContaminationRate(0.1),
                        random_state=42,
                    )

                # Time fitting
                start_time = time.time()
                adapter.fit(dataset)
                fit_time = time.time() - start_time

                # Time detection
                start_time = time.time()
                result = adapter.detect(dataset)
                detect_time = time.time() - start_time

                # Record results
                results.append(
                    {
                        "algorithm": algorithm,
                        "n_samples": n_samples,
                        "fit_time": fit_time,
                        "detect_time": detect_time,
                        "total_time": fit_time + detect_time,
                        "anomalies_detected": len(result.anomalies),
                        "anomaly_rate": len(result.anomalies) / n_samples,
                        "library": "sklearn",
                    }
                )

                print(
                    f"  ✓ Fit: {fit_time:.4f}s, Detect: {detect_time:.4f}s, Total: {fit_time + detect_time:.4f}s"
                )

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append(
                    {
                        "algorithm": algorithm,
                        "n_samples": n_samples,
                        "fit_time": None,
                        "detect_time": None,
                        "total_time": None,
                        "anomalies_detected": None,
                        "anomaly_rate": None,
                        "library": "sklearn",
                        "error": str(e),
                    }
                )

    return results


def benchmark_pyod_algorithms():
    """Benchmark PyOD-based algorithms."""
    print("\n🔍 Benchmarking PyOD Algorithms")
    print("=" * 50)

    # Test configurations
    data_sizes = [100, 500, 1000, 5000]
    algorithms = ["IForest", "LOF", "COPOD"]

    results = []

    for n_samples in data_sizes:
        dataset = generate_test_data(n_samples=n_samples, n_features=10)

        for algorithm in algorithms:
            print(f"Testing {algorithm} with {n_samples} samples...")

            try:
                # Create adapter
                if algorithm in ["LOF", "COPOD"]:
                    adapter = PyODAdapter(
                        algorithm_name=algorithm,
                        name=f"Benchmark {algorithm}",
                        contamination_rate=ContaminationRate(0.1),
                    )
                else:
                    adapter = PyODAdapter(
                        algorithm_name=algorithm,
                        name=f"Benchmark {algorithm}",
                        contamination_rate=ContaminationRate(0.1),
                        random_state=42,
                    )

                # Time fitting
                start_time = time.time()
                adapter.fit(dataset)
                fit_time = time.time() - start_time

                # Time detection
                start_time = time.time()
                result = adapter.detect(dataset)
                detect_time = time.time() - start_time

                # Record results
                results.append(
                    {
                        "algorithm": algorithm,
                        "n_samples": n_samples,
                        "fit_time": fit_time,
                        "detect_time": detect_time,
                        "total_time": fit_time + detect_time,
                        "anomalies_detected": len(result.anomalies),
                        "anomaly_rate": len(result.anomalies) / n_samples,
                        "library": "pyod",
                    }
                )

                print(
                    f"  ✓ Fit: {fit_time:.4f}s, Detect: {detect_time:.4f}s, Total: {fit_time + detect_time:.4f}s"
                )

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append(
                    {
                        "algorithm": algorithm,
                        "n_samples": n_samples,
                        "fit_time": None,
                        "detect_time": None,
                        "total_time": None,
                        "anomalies_detected": None,
                        "anomaly_rate": None,
                        "library": "pyod",
                        "error": str(e),
                    }
                )

    return results


def print_performance_summary(results):
    """Print performance summary."""
    print("\n📊 Performance Summary")
    print("=" * 80)

    successful_results = [r for r in results if r.get("total_time") is not None]
    failed_results = [r for r in results if r.get("total_time") is None]

    if successful_results:
        # Create results DataFrame
        df = pd.DataFrame(successful_results)

        print(f"✅ Successful runs: {len(successful_results)}")
        print(f"❌ Failed runs: {len(failed_results)}")

        # Performance by algorithm
        print("\n⚡ Performance by Algorithm:")
        print("-" * 40)

        for algorithm in df["algorithm"].unique():
            algo_data = df[df["algorithm"] == algorithm]
            avg_time = algo_data["total_time"].mean()
            max_samples = algo_data["n_samples"].max()

            print(f"{algorithm:20s}: {avg_time:.4f}s avg, max {max_samples} samples")

        # Performance by data size
        print("\n📏 Performance by Data Size:")
        print("-" * 40)

        for n_samples in sorted(df["n_samples"].unique()):
            size_data = df[df["n_samples"] == n_samples]
            avg_time = size_data["total_time"].mean()
            num_algorithms = len(size_data)

            print(
                f"{n_samples:6d} samples: {avg_time:.4f}s avg ({num_algorithms} algorithms)"
            )

        # Fastest algorithms
        print("\n🏃 Fastest Algorithms:")
        print("-" * 40)

        avg_times = (
            df.groupby(["algorithm", "library"])["total_time"].mean().sort_values()
        )
        for (algorithm, library), avg_time in avg_times.head(5).items():
            print(f"{algorithm} ({library}): {avg_time:.4f}s")

        # Accuracy analysis
        print("\n🎯 Anomaly Detection Rates:")
        print("-" * 40)

        avg_rates = (
            df.groupby(["algorithm", "library"])["anomaly_rate"]
            .mean()
            .sort_values(ascending=False)
        )
        for (algorithm, library), avg_rate in avg_rates.head(5).items():
            print(f"{algorithm} ({library}): {avg_rate:.2%}")

    if failed_results:
        print(f"\n❌ Failed Tests ({len(failed_results)}):")
        print("-" * 40)

        for result in failed_results:
            print(
                f"{result['algorithm']} ({result['library']}) - {result['n_samples']} samples: {result.get('error', 'Unknown error')}"
            )


def main():
    """Main benchmarking function."""
    print("🚀 Pynomaly Performance Benchmarking")
    print("=" * 60)

    all_results = []

    # Benchmark sklearn algorithms
    sklearn_results = benchmark_sklearn_algorithms()
    all_results.extend(sklearn_results)

    # Benchmark PyOD algorithms
    pyod_results = benchmark_pyod_algorithms()
    all_results.extend(pyod_results)

    # Print summary
    print_performance_summary(all_results)

    print("\n🎉 Benchmarking Complete!")

    return all_results


if __name__ == "__main__":
    results = main()
