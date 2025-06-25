"""Detection CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.presentation.cli.container import get_cli_container

app = typer.Typer()
console = Console()


@app.command("train")
def train_detector(
    detector: str = typer.Argument(..., help="Detector ID or name (can be partial)"),
    dataset: str = typer.Argument(..., help="Dataset ID or name (can be partial)"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate data before training"
    ),
    save_model: bool = typer.Option(
        True, "--save/--no-save", help="Save trained model"
    ),
):
    """Train a detector on a dataset."""
    container = get_cli_container()
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    train_use_case = container.train_detector_use_case()

    # Find detector
    detectors = detector_repo.find_all()
    matching_detectors = [
        d
        for d in detectors
        if str(d.id).startswith(detector) or d.name.lower().startswith(detector.lower())
    ]

    if not matching_detectors:
        console.print(f"[red]Error:[/red] No detector found matching '{detector}'")
        raise typer.Exit(1)

    if len(matching_detectors) > 1:
        console.print(f"[red]Error:[/red] Multiple detectors match '{detector}':")
        for d in matching_detectors:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    detector_obj = matching_detectors[0]

    # Find dataset
    datasets = dataset_repo.find_all()
    matching_datasets = [
        d
        for d in datasets
        if str(d.id).startswith(dataset) or d.name.lower().startswith(dataset.lower())
    ]

    if not matching_datasets:
        console.print(f"[red]Error:[/red] No dataset found matching '{dataset}'")
        raise typer.Exit(1)

    if len(matching_datasets) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset}':")
        for d in matching_datasets:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset_obj = matching_datasets[0]

    # Train detector
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training {detector_obj.algorithm}...", total=None)

        try:
            from pynomaly.application.use_cases import TrainDetectorRequest

            request = TrainDetectorRequest(
                detector_id=detector_obj.id,
                dataset=dataset_obj,
                validate_data=validate,
                save_model=save_model,
            )

            import asyncio

            response = asyncio.run(train_use_case.execute(request))

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]Error:[/red] Training failed: {str(e)}")
            raise typer.Exit(1)

    # Display results
    console.print("[green]✓[/green] Training completed!")
    console.print(f"\nDetector: {detector_obj.name}")
    console.print(f"Dataset: {dataset_obj.name}")
    console.print(f"Training time: {response.training_time_ms}ms")

    if response.dataset_summary:
        console.print("\nDataset summary:")
        for key, value in response.dataset_summary.items():
            console.print(f"  {key}: {value}")

    if response.validation_results:
        console.print("\nValidation results:")
        for key, value in response.validation_results.items():
            console.print(f"  {key}: {value}")


@app.command("run")
def detect_anomalies(
    detector: str = typer.Argument(..., help="Detector ID or name (can be partial)"),
    dataset: str = typer.Argument(..., help="Dataset ID or name (can be partial)"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate features"
    ),
    save_results: bool = typer.Option(
        True, "--save/--no-save", help="Save detection results"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Export results to file"
    ),
):
    """Run anomaly detection on a dataset."""
    container = get_cli_container()
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    detect_use_case = container.detect_anomalies_use_case()

    # Find detector
    detectors = detector_repo.find_all()
    matching_detectors = [
        d
        for d in detectors
        if str(d.id).startswith(detector) or d.name.lower().startswith(detector.lower())
    ]

    if not matching_detectors:
        console.print(f"[red]Error:[/red] No detector found matching '{detector}'")
        raise typer.Exit(1)

    if len(matching_detectors) > 1:
        console.print(f"[red]Error:[/red] Multiple detectors match '{detector}':")
        for d in matching_detectors:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    detector_obj = matching_detectors[0]

    if not detector_obj.is_fitted:
        console.print(
            f"[red]Error:[/red] Detector '{detector_obj.name}' is not trained"
        )
        console.print("Train the detector first using: pynomaly detect train")
        raise typer.Exit(1)

    # Find dataset
    datasets = dataset_repo.find_all()
    matching_datasets = [
        d
        for d in datasets
        if str(d.id).startswith(dataset) or d.name.lower().startswith(dataset.lower())
    ]

    if not matching_datasets:
        console.print(f"[red]Error:[/red] No dataset found matching '{dataset}'")
        raise typer.Exit(1)

    if len(matching_datasets) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset}':")
        for d in matching_datasets:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset_obj = matching_datasets[0]

    # Run detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting anomalies...", total=None)

        try:
            from pynomaly.application.use_cases import DetectAnomaliesRequest

            request = DetectAnomaliesRequest(
                detector_id=detector_obj.id,
                dataset=dataset_obj,
                validate_features=validate,
                save_results=save_results,
            )

            import asyncio

            response = asyncio.run(detect_use_case.execute(request))

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]Error:[/red] Detection failed: {str(e)}")
            raise typer.Exit(1)

    result = response.result

    # Display results
    console.print("[green]✓[/green] Detection completed!")
    console.print("\nResults:")
    console.print(f"  Total samples: {result.n_samples:,}")
    console.print(f"  Anomalies found: {result.n_anomalies:,}")
    console.print(f"  Anomaly rate: {result.anomaly_rate:.2%}")
    console.print(f"  Threshold: {result.threshold:.4f}")
    console.print(f"  Execution time: {result.execution_time_ms}ms")

    if result.score_statistics:
        console.print("\nScore statistics:")
        for key, value in result.score_statistics.items():
            console.print(f"  {key}: {value:.4f}")

    # Export results if requested
    if output:
        try:
            import pandas as pd

            # Get anomaly indices and scores
            anomaly_indices = response.anomaly_indices
            anomaly_scores = response.anomaly_scores

            # Create results dataframe
            results_df = pd.DataFrame(
                {
                    "index": range(len(anomaly_scores)),
                    "anomaly_score": anomaly_scores,
                    "is_anomaly": [
                        i in anomaly_indices for i in range(len(anomaly_scores))
                    ],
                }
            )

            # Save based on extension
            if output.suffix.lower() == ".csv":
                results_df.to_csv(output, index=False)
            elif output.suffix.lower() in [".parquet", ".pq"]:
                results_df.to_parquet(output, index=False)
            else:
                console.print(
                    "[yellow]Warning:[/yellow] Unknown format, saving as CSV"
                )
                results_df.to_csv(output, index=False)

            console.print(f"\n[green]✓[/green] Results exported to {output}")

        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to export results: {str(e)}")


@app.command("batch")
def batch_detect(
    detectors: list[str] = typer.Argument(..., help="Detector IDs or names"),
    dataset: str = typer.Argument(..., help="Dataset ID or name (can be partial)"),
    save_results: bool = typer.Option(
        True, "--save/--no-save", help="Save detection results"
    ),
):
    """Run multiple detectors on a dataset."""
    container = get_cli_container()
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    detection_service = container.detection_service()

    # Find detectors
    all_detectors = detector_repo.find_all()
    selected_detectors = []

    for detector_query in detectors:
        matching = [
            d
            for d in all_detectors
            if str(d.id).startswith(detector_query)
            or d.name.lower().startswith(detector_query.lower())
        ]

        if not matching:
            console.print(
                f"[yellow]Warning:[/yellow] No detector found matching '{detector_query}'"
            )
            continue

        if len(matching) > 1:
            console.print(
                f"[yellow]Warning:[/yellow] Multiple detectors match '{detector_query}', using first"
            )

        selected_detectors.append(matching[0])

    if not selected_detectors:
        console.print("[red]Error:[/red] No valid detectors found")
        raise typer.Exit(1)

    # Check all detectors are fitted
    unfitted = [d for d in selected_detectors if not d.is_fitted]
    if unfitted:
        console.print("[red]Error:[/red] The following detectors are not trained:")
        for d in unfitted:
            console.print(f"  - {d.name}")
        raise typer.Exit(1)

    # Find dataset
    datasets = dataset_repo.find_all()
    matching_datasets = [
        d
        for d in datasets
        if str(d.id).startswith(dataset) or d.name.lower().startswith(dataset.lower())
    ]

    if not matching_datasets:
        console.print(f"[red]Error:[/red] No dataset found matching '{dataset}'")
        raise typer.Exit(1)

    if len(matching_datasets) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset}':")
        for d in matching_datasets:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset_obj = matching_datasets[0]

    # Run batch detection
    console.print(
        f"Running {len(selected_detectors)} detectors on '{dataset_obj.name}'..."
    )

    try:
        import asyncio

        results = asyncio.run(
            detection_service.detect_with_multiple_detectors(
                detector_ids=[d.id for d in selected_detectors],
                dataset=dataset_obj,
                save_results=save_results,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] Batch detection failed: {str(e)}")
        raise typer.Exit(1)

    # Display results table
    table = Table(title="Batch Detection Results")
    table.add_column("Detector", style="cyan")
    table.add_column("Algorithm", style="yellow")
    table.add_column("Anomalies", style="red")
    table.add_column("Rate", style="magenta")
    table.add_column("Time (ms)", style="green")

    for detector in selected_detectors:
        if detector.id in results:
            result = results[detector.id]
            table.add_row(
                detector.name,
                detector.algorithm,
                f"{result.n_anomalies:,}",
                f"{result.anomaly_rate:.2%}",
                str(result.execution_time_ms),
            )

    console.print(table)


@app.command("evaluate")
def evaluate_detector(
    detector: str = typer.Argument(..., help="Detector ID or name (can be partial)"),
    dataset: str = typer.Argument(..., help="Dataset ID or name with labels"),
    cv: bool = typer.Option(False, "--cv", help="Use cross-validation"),
    folds: int = typer.Option(5, "--folds", "-k", help="Number of CV folds"),
    metrics: list[str] | None = typer.Option(
        None, "--metric", "-m", help="Metrics to compute"
    ),
):
    """Evaluate detector performance on labeled data."""
    container = get_cli_container()
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    evaluate_use_case = container.evaluate_model_use_case()

    # Find detector
    detectors = detector_repo.find_all()
    matching_detectors = [
        d
        for d in detectors
        if str(d.id).startswith(detector) or d.name.lower().startswith(detector.lower())
    ]

    if not matching_detectors:
        console.print(f"[red]Error:[/red] No detector found matching '{detector}'")
        raise typer.Exit(1)

    if len(matching_detectors) > 1:
        console.print(f"[red]Error:[/red] Multiple detectors match '{detector}':")
        for d in matching_detectors:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    detector_obj = matching_detectors[0]

    # Find dataset
    datasets = dataset_repo.find_all()
    matching_datasets = [
        d
        for d in datasets
        if str(d.id).startswith(dataset) or d.name.lower().startswith(dataset.lower())
    ]

    if not matching_datasets:
        console.print(f"[red]Error:[/red] No dataset found matching '{dataset}'")
        raise typer.Exit(1)

    if len(matching_datasets) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset}':")
        for d in matching_datasets:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset_obj = matching_datasets[0]

    if not dataset_obj.has_target:
        console.print(
            f"[red]Error:[/red] Dataset '{dataset_obj.name}' has no target labels"
        )
        console.print("Evaluation requires a labeled dataset")
        raise typer.Exit(1)

    # Run evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating detector...", total=None)

        try:
            from pynomaly.application.use_cases import EvaluateModelRequest

            request = EvaluateModelRequest(
                detector_id=detector_obj.id,
                test_dataset=dataset_obj,
                cross_validate=cv,
                n_folds=folds,
                metrics=metrics,
            )

            import asyncio

            response = asyncio.run(evaluate_use_case.execute(request))

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]Error:[/red] Evaluation failed: {str(e)}")
            raise typer.Exit(1)

    # Display results
    console.print("[green]✓[/green] Evaluation completed!")
    console.print(f"\nDetector: {detector_obj.name} ({detector_obj.algorithm})")
    console.print(f"Dataset: {dataset_obj.name}")

    # Metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, value in response.metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console.print(table)

    # Confusion matrix
    if response.confusion_matrix is not None:
        console.print("\nConfusion Matrix:")
        console.print(
            f"  TN: {response.confusion_matrix[0, 0]:,}  FP: {response.confusion_matrix[0, 1]:,}"
        )
        console.print(
            f"  FN: {response.confusion_matrix[1, 0]:,}  TP: {response.confusion_matrix[1, 1]:,}"
        )

    # Cross-validation results
    if response.cross_validation_scores:
        console.print("\nCross-Validation Results:")
        for metric, scores in response.cross_validation_scores.items():
            mean_score = sum(scores) / len(scores)
            std_score = (
                sum((s - mean_score) ** 2 for s in scores) / len(scores)
            ) ** 0.5
            console.print(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")


@app.command("results")
def list_results(
    detector: str | None = typer.Option(
        None, "--detector", "-d", help="Filter by detector"
    ),
    dataset: str | None = typer.Option(
        None, "--dataset", "-s", help="Filter by dataset"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
    latest: bool = typer.Option(False, "--latest", help="Show only the latest result"),
):
    """List detection results."""
    container = get_cli_container()
    result_repo = container.result_repository()
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()

    # Get results
    if latest:
        results = result_repo.find_recent(1)
    elif detector:
        # Find detector
        detectors = detector_repo.find_all()
        matching = [
            d
            for d in detectors
            if str(d.id).startswith(detector)
            or d.name.lower().startswith(detector.lower())
        ]
        if not matching:
            console.print(f"[red]Error:[/red] No detector found matching '{detector}'")
            raise typer.Exit(1)
        results = result_repo.find_by_detector(matching[0].id)
    elif dataset:
        # Find dataset
        datasets = dataset_repo.find_all()
        matching = [
            d
            for d in datasets
            if str(d.id).startswith(dataset)
            or d.name.lower().startswith(dataset.lower())
        ]
        if not matching:
            console.print(f"[red]Error:[/red] No dataset found matching '{dataset}'")
            raise typer.Exit(1)
        results = result_repo.find_by_dataset(matching[0].id)
    else:
        results = result_repo.find_recent(limit)

    if not results:
        console.print("No results found.")
        return

    # Create table
    table = Table(title="Detection Results")
    table.add_column("Time", style="dim")
    table.add_column("Detector", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("Samples", style="yellow")
    table.add_column("Anomalies", style="red")
    table.add_column("Rate", style="magenta")

    for result in results[:limit]:
        # Get detector and dataset names
        detector_obj = detector_repo.find_by_id(result.detector_id)
        dataset_obj = dataset_repo.find_by_id(result.dataset_id)

        table.add_row(
            result.timestamp.strftime("%Y-%m-%d %H:%M"),
            detector_obj.name if detector_obj else "Unknown",
            dataset_obj.name if dataset_obj else "Unknown",
            f"{result.n_samples:,}",
            f"{result.n_anomalies:,}",
            f"{result.anomaly_rate:.2%}",
        )

    console.print(table)
