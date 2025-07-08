"""Autonomous anomaly detection CLI commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pynomaly.application.services.autonomous_service import (
    AutonomousConfig,
    AutonomousDetectionService,
)
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
from pynomaly.infrastructure.data_loaders.excel_loader import ExcelLoader
from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader
from pynomaly.infrastructure.data_loaders.parquet_loader import ParquetLoader
from pynomaly.presentation.cli.container import get_cli_container

app = typer.Typer()
console = Console()


@app.command("detect")
def autonomous_detect(
    data_source: str = typer.Argument(
        ..., help="Path to data file or connection string"
    ),
    output: Path
    | None = typer.Option(None, "--output", "-o", help="Export results to file"),
    max_algorithms: int = typer.Option(
        5, "--max-algorithms", "-a", help="Maximum algorithms to try"
    ),
    confidence_threshold: float = typer.Option(
        0.8, "--confidence", "-c", help="Minimum confidence threshold"
    ),
    auto_tune: bool = typer.Option(
        True, "--auto-tune/--no-tune", help="Auto-tune hyperparameters"
    ),
    save_results: bool = typer.Option(
        True, "--save/--no-save", help="Save results to database"
    ),
    export_format: str = typer.Option(
        "csv", "--format", "-f", help="Export format (csv, parquet, excel)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    max_samples: int = typer.Option(
        10000, "--max-samples", help="Maximum samples for analysis"
    ),
    # Preprocessing options
    enable_preprocessing: bool = typer.Option(
        True, "--preprocess/--no-preprocess", help="Enable intelligent preprocessing"
    ),
    quality_threshold: float = typer.Option(
        0.8, "--quality-threshold", help="Data quality threshold for preprocessing"
    ),
    max_preprocessing_time: float = typer.Option(
        300.0, "--max-preprocess-time", help="Maximum preprocessing time (seconds)"
    ),
    preprocessing_strategy: str = typer.Option(
        "auto",
        "--preprocessing-strategy",
        help="Preprocessing strategy: auto, aggressive, conservative, minimal",
    ),
):
    """Run fully autonomous anomaly detection on any data source.

    This command automatically:
    - Detects data format and loads the data
    - Assesses data quality and applies intelligent preprocessing
    - Profiles the dataset to understand its characteristics
    - Recommends and configures optimal algorithms
    - Runs detection with the best algorithms
    - Provides comprehensive results and insights
    """

    # Create configuration
    config = AutonomousConfig(
        max_samples_analysis=max_samples,
        confidence_threshold=confidence_threshold,
        max_algorithms=max_algorithms,
        auto_tune_hyperparams=auto_tune,
        save_results=save_results,
        export_results=output is not None,
        export_format=export_format,
        verbose=verbose,
        # Preprocessing configuration
        enable_preprocessing=enable_preprocessing,
        quality_threshold=quality_threshold,
        max_preprocessing_time=max_preprocessing_time,
        preprocessing_strategy=preprocessing_strategy,
    )

    # Setup data loaders
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    # Get container dependencies
    container = get_cli_container()

    # Create autonomous service
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders,
    )

    console.print(
        Panel.fit(
            "[bold blue]🤖 Autonomous Anomaly Detection[/bold blue]\n"
            f"Data Source: {data_source}\n"
            f"Max Algorithms: {max_algorithms}\n"
            f"Auto-tune: {'Yes' if auto_tune else 'No'}\n"
            f"Preprocessing: {'Enabled' if enable_preprocessing else 'Disabled'}\n"
            f"Quality Threshold: {quality_threshold:.2f}",
            title="Configuration",
        )
    )

    # Run autonomous detection with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Add overall task
        main_task = progress.add_task("Running autonomous detection...", total=5)

        try:
            # Step 1: Data loading
            progress.update(main_task, description="🔍 Detecting and loading data...")

            # Step 2-5: Run autonomous detection
            results = asyncio.run(
                autonomous_service.detect_autonomous(data_source, config)
            )

            progress.update(
                main_task, completed=5, description="✅ Detection completed!"
            )

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

    # Display results
    _display_results(results, verbose)

    # Export results if requested
    if output:
        _export_autonomous_results(results, output, export_format)
        console.print(f"\n[green]✓[/green] Results exported to {output}")


@app.command("profile")
def profile_data(
    data_source: str = typer.Argument(..., help="Path to data file"),
    max_samples: int = typer.Option(
        10000, "--max-samples", help="Maximum samples for analysis"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Profile a dataset to understand its characteristics for anomaly detection."""

    # Setup data loaders
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    # Get container dependencies
    container = get_cli_container()

    # Create autonomous service
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders,
    )

    config = AutonomousConfig(max_samples_analysis=max_samples, verbose=verbose)

    console.print(
        Panel.fit(
            f"[bold blue]📊 Data Profiling[/bold blue]\nSource: {data_source}",
            title="Data Analysis",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing data...", total=None)

        try:
            # Load data
            dataset = asyncio.run(
                autonomous_service._auto_load_data(data_source, config)
            )

            # Profile data
            profile = asyncio.run(autonomous_service._profile_data(dataset, config))

            # Get recommendations
            recommendations = asyncio.run(
                autonomous_service._recommend_algorithms(profile, config)
            )

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

    # Display profile results
    _display_profile(profile, recommendations, verbose)


@app.command("quick")
def quick_detect(
    data_source: str = typer.Argument(..., help="Path to data file"),
    algorithm: str
    | None = typer.Option(None, "--algorithm", "-a", help="Force specific algorithm"),
    contamination: float
    | None = typer.Option(None, "--contamination", "-c", help="Contamination rate"),
    output: Path
    | None = typer.Option(None, "--output", "-o", help="Export results to file"),
):
    """Quick anomaly detection with minimal configuration.

    This is a simplified version that makes reasonable defaults for most use cases.
    """

    config = AutonomousConfig(
        max_algorithms=1 if algorithm else 3,
        auto_tune_hyperparams=False,  # Skip tuning for speed
        save_results=False,
        export_results=output is not None,
        verbose=False,
    )

    # Setup data loaders
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    # Get container dependencies
    container = get_cli_container()

    # Create autonomous service
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders,
    )

    console.print(f"🚀 Quick anomaly detection on {data_source}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running quick detection...", total=None)

        try:
            results = asyncio.run(
                autonomous_service.detect_autonomous(data_source, config)
            )

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

    # Display simplified results
    _display_quick_results(results)

    # Export if requested
    if output:
        _export_autonomous_results(results, output, "csv")
        console.print(f"Results saved to {output}")


def _display_results(results: dict, verbose: bool) -> None:
    """Display comprehensive autonomous detection results."""

    auto_results = results.get("autonomous_detection_results", {})

    if not auto_results.get("success"):
        console.print("[red]❌ Detection failed[/red]")
        return

    # Data profile summary
    profile = auto_results.get("data_profile", {})
    console.print("\n[bold blue]📊 Dataset Summary[/bold blue]")

    profile_table = Table(show_header=False, box=None)
    profile_table.add_column("Property", style="cyan")
    profile_table.add_column("Value", style="white")

    profile_table.add_row("Samples", f"{profile.get('samples', 0):,}")
    profile_table.add_row("Features", f"{profile.get('features', 0):,}")
    profile_table.add_row("Numeric Features", f"{profile.get('numeric_features', 0):,}")
    profile_table.add_row("Missing Data", f"{profile.get('missing_ratio', 0):.1%}")
    profile_table.add_row(
        "Complexity Score", f"{profile.get('complexity_score', 0):.2f}"
    )
    profile_table.add_row(
        "Recommended Contamination",
        f"{profile.get('recommended_contamination', 0):.1%}",
    )

    console.print(profile_table)

    # Algorithm recommendations
    recommendations = auto_results.get("algorithm_recommendations", [])
    if recommendations:
        console.print("\n[bold blue]🧠 Algorithm Recommendations[/bold blue]")

        rec_table = Table()
        rec_table.add_column("Algorithm", style="cyan")
        rec_table.add_column("Confidence", style="green")
        rec_table.add_column("Reasoning", style="white")

        for rec in recommendations:
            confidence = rec.get("confidence", 0)
            confidence_str = f"{confidence:.1%}"
            if confidence > 0.8:
                confidence_str = f"[green]{confidence_str}[/green]"
            elif confidence > 0.6:
                confidence_str = f"[yellow]{confidence_str}[/yellow]"
            else:
                confidence_str = f"[red]{confidence_str}[/red]"

            rec_table.add_row(
                rec.get("algorithm", "Unknown"),
                confidence_str,
                (
                    rec.get("reasoning", "")[:60] + "..."
                    if len(rec.get("reasoning", "")) > 60
                    else rec.get("reasoning", "")
                ),
            )

        console.print(rec_table)

    # Detection results
    detection_results = auto_results.get("detection_results", {})
    best_algorithm = auto_results.get("best_algorithm")

    if detection_results:
        console.print(
            f"\n[bold blue]🎯 Detection Results ({len(detection_results)} algorithms)[/bold blue]"
        )

        results_table = Table()
        results_table.add_column("Algorithm", style="cyan")
        results_table.add_column("Anomalies", style="red")
        results_table.add_column("Rate", style="magenta")
        results_table.add_column("Threshold", style="yellow")
        results_table.add_column("Time (ms)", style="green")
        results_table.add_column("Best", style="bold white")

        for algo, result in detection_results.items():
            is_best = algo == best_algorithm
            best_mark = "⭐" if is_best else ""

            results_table.add_row(
                f"[bold]{algo}[/bold]" if is_best else algo,
                f"{result.get('anomalies_found', 0):,}",
                f"{result.get('anomaly_rate', 0):.1%}",
                f"{result.get('threshold', 0):.4f}",
                f"{result.get('execution_time_ms', 0):,}",
                best_mark,
            )

        console.print(results_table)

    # Best result details
    best_result = auto_results.get("best_result")
    if best_result:
        console.print(
            f"\n[bold green]🏆 Best Result: {best_result.get('algorithm', 'Unknown')}[/bold green]"
        )

        summary = best_result.get("summary", {})
        console.print(
            f"Total Anomalies: [red]{summary.get('total_anomalies', 0):,}[/red]"
        )
        console.print(
            f"Anomaly Rate: [magenta]{summary.get('anomaly_rate', '0%')}[/magenta]"
        )
        console.print(
            f"Confidence: [green]{summary.get('confidence', 'Unknown')}[/green]"
        )

        # Show top anomalies if verbose
        if verbose and best_result.get("anomalies"):
            console.print("\n[bold blue]🚨 Top Anomalies[/bold blue]")

            anomaly_table = Table()
            anomaly_table.add_column("Rank", style="dim")
            anomaly_table.add_column("Index", style="cyan")
            anomaly_table.add_column("Score", style="red")
            anomaly_table.add_column("Confidence", style="green")

            for i, anomaly in enumerate(best_result["anomalies"][:10]):  # Top 10
                anomaly_table.add_row(
                    str(i + 1),
                    str(anomaly.get("index", "Unknown")),
                    f"{anomaly.get('score', 0):.4f}",
                    str(anomaly.get("confidence", "N/A")),
                )

            console.print(anomaly_table)


def _display_profile(profile, recommendations, verbose: bool) -> None:
    """Display data profiling results."""

    console.print("\n[bold blue]📊 Dataset Profile[/bold blue]")

    # Basic statistics
    basic_table = Table(title="Basic Statistics")
    basic_table.add_column("Property", style="cyan")
    basic_table.add_column("Value", style="white")

    basic_table.add_row("Samples", f"{profile.n_samples:,}")
    basic_table.add_row("Features", f"{profile.n_features:,}")
    basic_table.add_row("Numeric Features", f"{profile.numeric_features:,}")
    basic_table.add_row("Categorical Features", f"{profile.categorical_features:,}")
    basic_table.add_row("Temporal Features", f"{profile.temporal_features:,}")
    basic_table.add_row("Missing Values", f"{profile.missing_values_ratio:.1%}")
    basic_table.add_row("Sparsity Ratio", f"{profile.sparsity_ratio:.1%}")

    console.print(basic_table)

    # Analysis results
    analysis_table = Table(title="Data Analysis")
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="white")

    analysis_table.add_row("Correlation Score", f"{profile.correlation_score:.3f}")
    analysis_table.add_row("Complexity Score", f"{profile.complexity_score:.3f}")
    analysis_table.add_row("Outlier Estimate", f"{profile.outlier_ratio_estimate:.1%}")
    analysis_table.add_row(
        "Recommended Contamination", f"{profile.recommended_contamination:.1%}"
    )
    analysis_table.add_row(
        "Seasonality Detected", "Yes" if profile.seasonality_detected else "No"
    )
    analysis_table.add_row("Trend Detected", "Yes" if profile.trend_detected else "No")

    console.print(analysis_table)

    # Data Quality and Preprocessing Information
    if hasattr(profile, "quality_score") and profile.quality_score is not None:
        console.print("\n[bold blue]🔧 Data Quality & Preprocessing[/bold blue]")

        quality_table = Table(title="Data Quality Assessment")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="white")

        # Quality score with color coding
        quality_score = profile.quality_score
        quality_color = (
            "green"
            if quality_score >= 0.8
            else "yellow"
            if quality_score >= 0.6
            else "red"
        )
        quality_table.add_row(
            "Quality Score", f"[{quality_color}]{quality_score:.2f}[/{quality_color}]"
        )

        if hasattr(profile, "preprocessing_recommended"):
            quality_table.add_row(
                "Preprocessing Recommended",
                "Yes" if profile.preprocessing_recommended else "No",
            )

        if hasattr(profile, "preprocessing_applied"):
            quality_table.add_row(
                "Preprocessing Applied",
                "Yes" if profile.preprocessing_applied else "No",
            )

        console.print(quality_table)

        # Show preprocessing details if applied
        if (
            hasattr(profile, "preprocessing_metadata")
            and profile.preprocessing_metadata
        ):
            metadata = profile.preprocessing_metadata
            if metadata.get("preprocessing_applied"):
                console.print("\n[bold green]✓ Preprocessing Applied[/bold green]")

                if "applied_steps" in metadata:
                    steps_table = Table(title="Applied Preprocessing Steps")
                    steps_table.add_column("Step", style="cyan")
                    steps_table.add_column("Action", style="green")
                    steps_table.add_column("Details", style="white")

                    for step in metadata["applied_steps"]:
                        step_type = step.get("type", "Unknown")
                        action = step.get("action", "Unknown")

                        # Format details based on step type
                        if "columns" in step:
                            columns = step["columns"]
                            if isinstance(columns, list):
                                details = f"{len(columns)} columns"
                                if len(columns) <= 3:
                                    details += f": {', '.join(columns)}"
                            else:
                                details = str(columns)
                        elif "count" in step:
                            details = f"{step['count']} items"
                        else:
                            details = ""

                        steps_table.add_row(
                            step_type.replace("_", " ").title(), action.title(), details
                        )

                    console.print(steps_table)

                # Show shape changes
                original_shape = metadata.get("original_shape")
                final_shape = metadata.get("final_shape")
                if original_shape and final_shape:
                    console.print(
                        f"\n[dim]Shape: {original_shape[0]:,}×{original_shape[1]} → {final_shape[0]:,}×{final_shape[1]}[/dim]"
                    )

        # Show quality issues if available
        if (
            hasattr(profile, "quality_report")
            and profile.quality_report
            and profile.quality_report.issues
        ):
            console.print(
                "\n[bold yellow]⚠️ Data Quality Issues Detected[/bold yellow]"
            )

            for issue in profile.quality_report.issues[:5]:  # Show top 5 issues
                severity_color = (
                    "red"
                    if issue.severity > 0.7
                    else "yellow"
                    if issue.severity > 0.4
                    else "green"
                )
                console.print(
                    f"• [{severity_color}]{issue.issue_type.value.replace('_', ' ').title()}[/{severity_color}]: {issue.description}"
                )

            if len(profile.quality_report.issues) > 5:
                console.print(
                    f"[dim]... and {len(profile.quality_report.issues) - 5} more issues[/dim]"
                )

    # Algorithm recommendations
    if recommendations:
        console.print("\n[bold blue]🧠 Recommended Algorithms[/bold blue]")

        for i, rec in enumerate(recommendations, 1):
            confidence_color = (
                "green"
                if rec.confidence > 0.8
                else "yellow"
                if rec.confidence > 0.6
                else "red"
            )

            console.print(
                Panel(
                    f"[bold]{rec.algorithm}[/bold]\n"
                    f"Confidence: [{confidence_color}]{rec.confidence:.1%}[/{confidence_color}]\n"
                    f"Reasoning: {rec.reasoning}\n"
                    f"Expected Performance: {rec.expected_performance:.1%}",
                    title=f"#{i}",
                )
            )


def _display_quick_results(results: dict) -> None:
    """Display simplified results for quick detection."""

    auto_results = results.get("autonomous_detection_results", {})

    if not auto_results.get("success"):
        console.print("[red]❌ Detection failed[/red]")
        return

    best_result = auto_results.get("best_result")
    if best_result:
        summary = best_result.get("summary", {})
        console.print("\n✅ [green]Detection Complete[/green]")
        console.print(
            f"Algorithm: [cyan]{best_result.get('algorithm', 'Unknown')}[/cyan]"
        )
        console.print(
            f"Anomalies Found: [red]{summary.get('total_anomalies', 0):,}[/red]"
        )
        console.print(
            f"Anomaly Rate: [magenta]{summary.get('anomaly_rate', '0%')}[/magenta]"
        )
        console.print(
            f"Confidence: [green]{summary.get('confidence', 'Unknown')}[/green]"
        )


def _export_autonomous_results(
    results: dict, output_path: Path, format_type: str
) -> None:
    """Export autonomous detection results."""

    auto_results = results.get("autonomous_detection_results", {})
    best_result = auto_results.get("best_result")

    if not best_result:
        console.print("[yellow]Warning:[/yellow] No results to export")
        return

    # Create export data
    export_data = {
        "metadata": {
            "detection_type": "autonomous",
            "best_algorithm": best_result.get("algorithm"),
            "data_profile": auto_results.get("data_profile"),
            "algorithm_recommendations": auto_results.get("algorithm_recommendations"),
            "detection_results": auto_results.get("detection_results"),
        },
        "anomalies": best_result.get("anomalies", []),
        "summary": best_result.get("summary", {}),
    }

    # Export based on format
    if format_type.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
    else:
        # Convert to DataFrame for CSV/Excel export
        import pandas as pd

        # Create anomalies DataFrame
        if export_data["anomalies"]:
            anomalies_df = pd.DataFrame(export_data["anomalies"])
        else:
            anomalies_df = pd.DataFrame()

        # Add metadata as additional rows/columns
        anomalies_df["detection_algorithm"] = best_result.get("algorithm", "Unknown")
        anomalies_df["anomaly_rate"] = best_result.get("summary", {}).get(
            "anomaly_rate", "0%"
        )

        if format_type.lower() == "csv":
            anomalies_df.to_csv(output_path, index=False)
        elif format_type.lower() == "excel":
            anomalies_df.to_excel(output_path, index=False)
        elif format_type.lower() == "parquet":
            anomalies_df.to_parquet(output_path, index=False)
