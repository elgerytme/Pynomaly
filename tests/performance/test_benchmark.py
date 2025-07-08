#!/usr/bin/env python
"""Benchmark different anomaly detection algorithms."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import pandas as pd
from pynomaly.application.use_cases import (
    DetectAnomaliesRequest,
    EvaluateModelRequest,
    TrainDetectorRequest,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config import create_container
from rich.console import Console
from rich.table import Table

console = Console()


async def benchmark_algorithm(
    container, algorithm: str, dataset: Dataset, contamination: float = 0.1
) -> dict[str, float]:
    """Benchmark a single algorithm."""
    # Create detector
    detector = Detector(
        name=f"Benchmark {algorithm}",
        algorithm=algorithm,
        parameters={"contamination": contamination},
    )

    detector_repo = container.detector_repository()
    detector_repo.save(detector)

    # Train detector
    train_use_case = container.train_detector_use_case()
    train_request = TrainDetectorRequest(
        detector_id=detector.id,
        dataset=dataset,
        validate_data=False,  # Skip validation for speed
        save_model=True,
    )

    train_start = time.time()
    await train_use_case.execute(train_request)
    train_time = time.time() - train_start

    # Detect anomalies
    detect_use_case = container.detect_anomalies_use_case()
    detect_request = DetectAnomaliesRequest(
        detector_id=detector.id,
        dataset=dataset,
        validate_features=False,
        save_results=True,
    )

    detect_start = time.time()
    detect_response = await detect_use_case.execute(detect_request)
    detect_time = time.time() - detect_start

    # Evaluate if labels available
    metrics = {}
    if dataset.has_target:
        evaluate_use_case = container.evaluate_model_use_case()
        eval_request = EvaluateModelRequest(
            detector_id=detector.id,
            test_dataset=dataset,
            cross_validate=False,
            metrics=["precision", "recall", "f1", "roc_auc"],
        )

        eval_response = await evaluate_use_case.execute(eval_request)
        metrics = eval_response.metrics

    return {
        "algorithm": algorithm,
        "train_time": train_time,
        "detect_time": detect_time,
        "total_time": train_time + detect_time,
        "n_anomalies": detect_response.result.n_anomalies,
        "anomaly_rate": detect_response.result.anomaly_rate,
        **metrics,
    }


async def run_benchmark(
    algorithms: list[str],
    dataset_path: Path,
    contamination: float = 0.1,
    sample_size: int = None,
) -> list[dict[str, float]]:
    """Run benchmark on multiple algorithms."""
    container = create_container()

    # Load dataset
    console.print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        console.print(f"Sampled {sample_size} rows")

    # Check for target column
    target_column = None
    if "label" in df.columns:
        target_column = "label"
    elif "target" in df.columns:
        target_column = "target"

    dataset = Dataset(
        name="Benchmark Dataset",
        data=df.drop(columns=[target_column] if target_column else []),
        target_column=target_column,
    )

    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    console.print(f"Dataset shape: {dataset.shape}")
    console.print(f"Has labels: {dataset.has_target}\n")

    # Run benchmarks
    results = []

    with console.status("Running benchmarks...") as status:
        for algo in algorithms:
            status.update(f"Benchmarking {algo}...")
            try:
                result = await benchmark_algorithm(
                    container, algo, dataset, contamination
                )
                results.append(result)
                console.print(f"✓ {algo} completed")
            except Exception as e:
                console.print(f"✗ {algo} failed: {str(e)}")

    return results


def display_results(results: list[dict[str, float]]):
    """Display benchmark results in a table."""
    if not results:
        console.print("No results to display")
        return

    # Create table
    table = Table(title="Anomaly Detection Benchmark Results")

    # Add columns
    table.add_column("Algorithm", style="cyan")
    table.add_column("Train (s)", style="yellow")
    table.add_column("Detect (s)", style="yellow")
    table.add_column("Total (s)", style="green")
    table.add_column("Anomalies", style="red")
    table.add_column("Rate (%)", style="magenta")

    # Add metric columns if available
    if "precision" in results[0]:
        table.add_column("Precision", style="blue")
        table.add_column("Recall", style="blue")
        table.add_column("F1", style="blue")
        table.add_column("ROC-AUC", style="blue")

    # Sort by total time
    results.sort(key=lambda x: x["total_time"])

    # Add rows
    for result in results:
        row = [
            result["algorithm"],
            f"{result['train_time']:.3f}",
            f"{result['detect_time']:.3f}",
            f"{result['total_time']:.3f}",
            str(result["n_anomalies"]),
            f"{result['anomaly_rate'] * 100:.1f}",
        ]

        if "precision" in result:
            row.extend(
                [
                    f"{result.get('precision', 0):.3f}",
                    f"{result.get('recall', 0):.3f}",
                    f"{result.get('f1', 0):.3f}",
                    f"{result.get('roc_auc', 0):.3f}",
                ]
            )

        table.add_row(*row)

    console.print(table)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("benchmark_results.csv", index=False)
    console.print("\nResults saved to benchmark_results.csv")


def main():
    """Run benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark anomaly detection algorithms"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset CSV file")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Algorithms to benchmark (default: common algorithms)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected contamination rate (default: 0.1)",
    )
    parser.add_argument(
        "--sample-size", type=int, help="Sample size for large datasets"
    )
    parser.add_argument(
        "--all", action="store_true", help="Benchmark all available algorithms"
    )

    args = parser.parse_args()

    # Determine algorithms to benchmark
    if args.all:
        container = create_container()
        pyod_adapter = container.pyod_adapter()
        algorithms = pyod_adapter.list_algorithms()
    elif args.algorithms:
        algorithms = args.algorithms
    else:
        # Default algorithms
        algorithms = [
            "IsolationForest",
            "LOF",
            "OCSVM",
            "HBOS",
            "KNN",
            "ABOD",
            "COPOD",
            "ECOD",
            "MCD",
            "PCA",
        ]

    console.print(f"Benchmarking {len(algorithms)} algorithms\n")

    # Run benchmark
    results = asyncio.run(
        run_benchmark(algorithms, args.dataset, args.contamination, args.sample_size)
    )

    # Display results
    display_results(results)


if __name__ == "__main__":
    main()
