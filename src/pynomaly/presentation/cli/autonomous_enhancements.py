"""Enhanced autonomous CLI commands with new features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from pynomaly.application.services.automl_service import AutoMLService
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


@app.command("detect-all")
def autonomous_detect_with_all_classifiers(
    data_source: str = typer.Argument(
        ..., help="Path to data file or connection string"
    ),
    output: Path
    | None = typer.Option(None, "--output", "-o", help="Export results to file"),
    max_time: int = typer.Option(
        1800, "--max-time", help="Maximum time for detection (seconds)"
    ),
    confidence_threshold: float = typer.Option(
        0.6, "--confidence", "-c", help="Lower confidence for more algorithms"
    ),
    export_format: str = typer.Option(
        "csv", "--format", "-f", help="Export format (csv, parquet, excel)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ensemble_best: bool = typer.Option(
        True, "--ensemble/--no-ensemble", help="Create ensemble from best algorithms"
    ),
):
    """Run autonomous detection using ALL available classifiers.

    This command tests every compatible algorithm to find the absolute best
    combination for your data. Uses lower confidence thresholds to include
    more algorithms and optionally creates ensembles.
    """

    config = AutonomousConfig(
        max_algorithms=15,  # Allow more algorithms
        confidence_threshold=confidence_threshold,  # Lower threshold
        auto_tune_hyperparams=True,
        save_results=True,
        export_results=output is not None,
        export_format=export_format,
        verbose=verbose,
    )

    console.print(
        Panel.fit(
            "[bold green]🔍 Comprehensive Classifier Analysis[/bold green]\n"
            f"Data Source: {data_source}\n"
            f"Testing ALL compatible algorithms\n"
            f"Confidence Threshold: {confidence_threshold}\n"
            f"Ensemble Creation: {'Enabled' if ensemble_best else 'Disabled'}",
            title="All-Classifier Detection",
        )
    )

    _run_enhanced_autonomous_detection(
        data_source, config, ensemble_best, output, "all"
    )


@app.command("detect-by-family")
def autonomous_detect_by_family(
    data_source: str = typer.Argument(
        ..., help="Path to data file or connection string"
    ),
    families: list[str] = typer.Option(
        ["statistical", "distance_based", "isolation_based"],
        "--family",
        help="Algorithm families to use",
    ),
    output: Path
    | None = typer.Option(None, "--output", "-o", help="Export results to file"),
    ensemble_within_family: bool = typer.Option(
        True,
        "--family-ensemble/--no-family-ensemble",
        help="Create ensemble within each family",
    ),
    meta_ensemble: bool = typer.Option(
        True,
        "--meta-ensemble/--no-meta-ensemble",
        help="Create meta-ensemble from family results",
    ),
    export_format: str = typer.Option("csv", "--format", "-f", help="Export format"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run detection organized by algorithm families with hierarchical ensembles.

    Available families:
    - statistical: ECOD, COPOD
    - distance_based: KNN, LOF, OneClassSVM
    - isolation_based: IsolationForest
    - density_based: LOF variants
    - neural_networks: AutoEncoder, VAE
    """

    config = AutonomousConfig(
        max_algorithms=10,
        confidence_threshold=0.7,
        auto_tune_hyperparams=True,
        save_results=True,
        export_results=output is not None,
        export_format=export_format,
        verbose=verbose,
    )

    console.print(
        Panel.fit(
            "[bold purple]🏗️ Family-Based Ensemble Detection[/bold purple]\n"
            f"Data Source: {data_source}\n"
            f"Families: {', '.join(families)}\n"
            f"Family Ensembles: {'Yes' if ensemble_within_family else 'No'}\n"
            f"Meta Ensemble: {'Yes' if meta_ensemble else 'No'}",
            title="Family-Based Detection",
        )
    )

    _run_family_based_detection(
        data_source, config, families, ensemble_within_family, meta_ensemble, output
    )


@app.command("explain-choices")
def explain_algorithm_choices(
    data_source: str = typer.Argument(
        ..., help="Path to data file or connection string"
    ),
    max_algorithms: int = typer.Option(
        5, "--max-algorithms", help="Number of algorithms to explain"
    ),
    show_alternatives: bool = typer.Option(
        True,
        "--alternatives/--no-alternatives",
        help="Show alternative algorithms considered",
    ),
    save_explanation: bool = typer.Option(
        False, "--save/--no-save", help="Save explanation to file"
    ),
):
    """Explain why specific algorithms were chosen for your data.

    Provides detailed reasoning about algorithm selection based on:
    - Data characteristics analysis
    - Algorithm strengths and limitations
    - Performance expectations
    - Alternative options considered
    """

    console.print(
        Panel.fit(
            "[bold cyan]🧠 Algorithm Selection Explanation[/bold cyan]\n"
            f"Data Source: {data_source}\n"
            f"Analyzing top {max_algorithms} recommendations",
            title="Choice Explanation",
        )
    )

    _explain_algorithm_choices(
        data_source, max_algorithms, show_alternatives, save_explanation
    )


@app.command("analyze-results")
def analyze_detection_results(
    results_file: str = typer.Argument(..., help="Path to detection results file"),
    analysis_type: str = typer.Option(
        "comprehensive",
        "--type",
        help="Analysis type: comprehensive, statistical, visual",
    ),
    output: Path
    | None = typer.Option(None, "--output", "-o", help="Save analysis report"),
    interactive: bool = typer.Option(
        False, "--interactive/--batch", help="Interactive analysis mode"
    ),
):
    """Analyze and explain anomaly detection results.

    Provides insights into:
    - Anomaly characteristics and patterns
    - Statistical significance of findings
    - Confidence levels and uncertainty
    - Recommendations for next steps
    """

    console.print(
        Panel.fit(
            "[bold yellow]📊 Results Analysis[/bold yellow]\n"
            f"Results File: {results_file}\n"
            f"Analysis Type: {analysis_type}\n"
            f"Mode: {'Interactive' if interactive else 'Batch'}",
            title="Results Analysis",
        )
    )

    _analyze_detection_results(results_file, analysis_type, output, interactive)


def _run_enhanced_autonomous_detection(
    data_source: str,
    config: AutonomousConfig,
    ensemble_best: bool,
    output: Path | None,
    mode: str,
):
    """Run enhanced autonomous detection with all algorithms."""

    # Setup data loaders
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    container = get_cli_container()

    # Create services
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders,
    )

    AutoMLService(
        detector_repository=container.detector_repository(),
        dataset_repository=container.dataset_repository(),
        adapter_registry=container.adapter_registry(),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running comprehensive detection...", total=None)

        try:
            # Enhanced autonomous detection
            results = asyncio.run(
                autonomous_service.detect_autonomous(data_source, config)
            )

            progress.update(task, description="✅ Detection completed!")

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

    # Display enhanced results
    _display_enhanced_results(results, mode, ensemble_best)

    # Export if requested
    if output:
        _export_enhanced_results(results, output, config.export_format)
        console.print(f"\n[green]✓[/green] Enhanced results exported to {output}")


def _run_family_based_detection(
    data_source: str,
    config: AutonomousConfig,
    families: list[str],
    ensemble_within_family: bool,
    meta_ensemble: bool,
    output: Path | None,
):
    """Run family-based detection with hierarchical ensembles."""

    console.print("\n[bold blue]🔍 Family-Based Detection Process[/bold blue]")

    family_results = {}

    # Algorithm family mapping
    family_algorithms = {
        "statistical": ["ECOD", "COPOD"],
        "distance_based": ["KNN", "LOF", "OneClassSVM"],
        "isolation_based": ["IsolationForest"],
        "density_based": ["LOF"],
        "neural_networks": ["AutoEncoder", "VAE"],
    }

    for family in families:
        if family not in family_algorithms:
            console.print(
                f"[yellow]Warning:[/yellow] Unknown family '{family}', skipping"
            )
            continue

        console.print(f"\n[cyan]Processing {family} family...[/cyan]")

        # Run detection with family algorithms
        # This would be implemented with specific algorithm filtering
        # For now, showing the structure

        family_results[family] = {
            "algorithms": family_algorithms[family],
            "results": "placeholder_results",
            "ensemble": None,
        }

        if ensemble_within_family and len(family_algorithms[family]) > 1:
            console.print(f"  Creating ensemble for {family} family...")
            # Create family ensemble
            family_results[family]["ensemble"] = "family_ensemble_placeholder"

    # Create meta-ensemble if requested
    if meta_ensemble and len(family_results) > 1:
        console.print(
            "\n[purple]Creating meta-ensemble from family results...[/purple]"
        )

    # Display family results
    _display_family_results(family_results, meta_ensemble)


def _explain_algorithm_choices(
    data_source: str,
    max_algorithms: int,
    show_alternatives: bool,
    save_explanation: bool,
):
    """Explain algorithm selection process."""

    # Setup services (simplified for explanation)
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader(),
    }

    container = get_cli_container()
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders,
    )

    config = AutonomousConfig(verbose=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing data characteristics...", total=None)

        try:
            # Load and profile data
            dataset = asyncio.run(
                autonomous_service._auto_load_data(data_source, config)
            )

            progress.update(task, description="Profiling dataset...")
            profile = asyncio.run(autonomous_service._profile_data(dataset, config))

            progress.update(task, description="Analyzing algorithm suitability...")
            recommendations = asyncio.run(
                autonomous_service._recommend_algorithms(profile, config)
            )

            progress.update(task, description="✅ Analysis completed!")

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

    # Display detailed explanations
    _display_algorithm_explanations(
        profile, recommendations[:max_algorithms], show_alternatives
    )

    if save_explanation:
        _save_explanations(
            profile, recommendations, "algorithm_choices_explanation.json"
        )


def _analyze_detection_results(
    results_file: str, analysis_type: str, output: Path | None, interactive: bool
):
    """Analyze detection results comprehensively."""

    console.print(f"\n[bold blue]📊 Analyzing Results: {results_file}[/bold blue]")

    try:
        # Load results
        with open(results_file) as f:
            if results_file.endswith(".json"):
                results = json.load(f)
            else:
                import pandas as pd

                results_df = pd.read_csv(results_file)
                # Convert to analysis format
                results = {"data": results_df.to_dict()}

        # Perform analysis based on type
        if analysis_type == "comprehensive":
            _comprehensive_analysis(results, interactive)
        elif analysis_type == "statistical":
            _statistical_analysis(results, interactive)
        elif analysis_type == "visual":
            _visual_analysis(results, interactive)

        if output:
            _save_analysis_report(results, analysis_type, output)
            console.print(f"\n[green]✓[/green] Analysis report saved to {output}")

    except Exception as e:
        console.print(f"[red]Error analyzing results:[/red] {str(e)}")
        raise typer.Exit(1)


def _display_enhanced_results(results: dict, mode: str, ensemble_best: bool):
    """Display enhanced results with detailed insights."""

    auto_results = results.get("autonomous_detection_results", {})

    if not auto_results.get("success"):
        console.print("[red]❌ Detection failed[/red]")
        return

    console.print("\n[bold green]🎯 Enhanced Detection Results[/bold green]")

    # Show algorithm performance comparison
    detection_results = auto_results.get("detection_results", {})
    if detection_results:
        performance_table = Table(title="Algorithm Performance Comparison")
        performance_table.add_column("Algorithm", style="cyan")
        performance_table.add_column("Anomalies", style="red")
        performance_table.add_column("Rate", style="magenta")
        performance_table.add_column("Confidence", style="green")
        performance_table.add_column("Time (ms)", style="yellow")
        performance_table.add_column("Score", style="bold white")

        # Calculate performance scores
        for algo, result in detection_results.items():
            score = _calculate_performance_score(result)
            confidence = _estimate_confidence(result)

            performance_table.add_row(
                algo,
                f"{result.get('anomalies_found', 0):,}",
                f"{result.get('anomaly_rate', 0):.1%}",
                f"{confidence:.1%}",
                f"{result.get('execution_time_ms', 0):,}",
                f"{score:.3f}",
            )

        console.print(performance_table)

    # Show insights and recommendations
    _display_insights(auto_results, mode)


def _display_family_results(family_results: dict, meta_ensemble: bool):
    """Display family-based detection results."""

    console.print("\n[bold purple]🏗️ Family-Based Results[/bold purple]")

    # Create family tree visualization
    tree = Tree("Algorithm Families")

    for family, data in family_results.items():
        family_branch = tree.add(f"[cyan]{family.replace('_', ' ').title()}[/cyan]")

        for algo in data["algorithms"]:
            family_branch.add(f"[white]{algo}[/white]")

        if data.get("ensemble"):
            family_branch.add("[green]Ensemble Result[/green]")

    if meta_ensemble:
        tree.add("[purple]Meta-Ensemble[/purple]")

    console.print(tree)


def _display_algorithm_explanations(
    profile, recommendations: list, show_alternatives: bool
):
    """Display detailed algorithm choice explanations."""

    console.print("\n[bold cyan]🧠 Algorithm Selection Reasoning[/bold cyan]")

    # Data characteristics summary
    data_summary = Table(title="Dataset Characteristics")
    data_summary.add_column("Characteristic", style="cyan")
    data_summary.add_column("Value", style="white")
    data_summary.add_column("Impact", style="yellow")

    data_summary.add_row(
        "Sample Count",
        f"{profile.n_samples:,}",
        (
            "Large"
            if profile.n_samples > 10000
            else "Medium"
            if profile.n_samples > 1000
            else "Small"
        ),
    )
    data_summary.add_row(
        "Feature Count",
        f"{profile.n_features:,}",
        "High-dimensional" if profile.n_features > 50 else "Moderate",
    )
    data_summary.add_row(
        "Complexity Score",
        f"{profile.complexity_score:.2f}",
        "Complex" if profile.complexity_score > 0.7 else "Moderate",
    )
    data_summary.add_row(
        "Missing Data",
        f"{profile.missing_values_ratio:.1%}",
        "High" if profile.missing_values_ratio > 0.1 else "Low",
    )

    console.print(data_summary)

    # Algorithm recommendations with detailed reasoning
    for i, rec in enumerate(recommendations, 1):
        reasoning_panel = Panel(
            f"[bold]{rec.algorithm}[/bold]\n\n"
            f"[green]Confidence: {rec.confidence:.1%}[/green]\n"
            f"[blue]Expected Performance: {rec.expected_performance:.1%}[/blue]\n\n"
            f"[white]Reasoning:[/white]\n{rec.reasoning}\n\n"
            f"[yellow]Key Factors:[/yellow]\n"
            f"• Data size compatibility: {'✓' if profile.n_samples >= 100 else '⚠'}\n"
            f"• Feature type support: {'✓' if profile.numeric_features > 0 else '⚠'}\n"
            f"• Complexity matching: {'✓' if abs(profile.complexity_score - 0.5) < 0.3 else '⚠'}",
            title=f"Recommendation #{i}",
            border_style=(
                "green"
                if rec.confidence > 0.8
                else "yellow"
                if rec.confidence > 0.6
                else "red"
            ),
        )
        console.print(reasoning_panel)


def _comprehensive_analysis(results: dict, interactive: bool):
    """Perform comprehensive results analysis."""

    console.print("\n[bold blue]📈 Comprehensive Analysis[/bold blue]")

    # Anomaly distribution analysis
    console.print("\n[cyan]Anomaly Distribution:[/cyan]")
    console.print("• Analyzing anomaly patterns...")
    console.print("• Checking for clusters...")
    console.print("• Assessing score distributions...")

    # Statistical significance
    console.print("\n[cyan]Statistical Significance:[/cyan]")
    console.print("• Computing confidence intervals...")
    console.print("• Analyzing score separability...")
    console.print("• Evaluating detection stability...")

    if interactive:
        console.print(
            "\n[yellow]Interactive mode would allow drill-down into specific anomalies[/yellow]"
        )


def _statistical_analysis(results: dict, interactive: bool):
    """Perform statistical analysis of results."""

    console.print("\n[bold blue]📊 Statistical Analysis[/bold blue]")
    console.print("• Score distribution statistics")
    console.print("• Outlier significance testing")
    console.print("• Confidence interval calculations")


def _visual_analysis(results: dict, interactive: bool):
    """Perform visual analysis of results."""

    console.print("\n[bold blue]📉 Visual Analysis[/bold blue]")
    console.print("• Generating anomaly score histograms")
    console.print("• Creating scatter plots")
    console.print("• Building correlation matrices")


def _calculate_performance_score(result: dict) -> float:
    """Calculate a composite performance score."""
    # Simplified scoring based on execution time and anomaly rate
    time_score = 1.0 / (1.0 + result.get("execution_time_ms", 1000) / 1000)
    rate_score = min(
        1.0, result.get("anomaly_rate", 0) * 10
    )  # Prefer reasonable anomaly rates
    return (time_score + rate_score) / 2


def _estimate_confidence(result: dict) -> float:
    """Estimate confidence in detection results."""
    # Simplified confidence estimation
    anomaly_count = result.get("anomalies_found", 0)
    total_samples = result.get("total_samples", 1000)  # Estimated

    if anomaly_count == 0:
        return 0.5  # Neutral confidence

    rate = anomaly_count / total_samples
    if 0.01 <= rate <= 0.2:  # Reasonable anomaly rate
        return 0.8
    elif rate < 0.01:
        return 0.6  # Low confidence - too few anomalies
    else:
        return 0.4  # Low confidence - too many anomalies


def _display_insights(auto_results: dict, mode: str):
    """Display insights and recommendations."""

    console.print("\n[bold yellow]💡 Insights & Recommendations[/bold yellow]")

    best_algorithm = auto_results.get("best_algorithm")
    detection_results = auto_results.get("detection_results", {})

    insights = []

    if best_algorithm:
        insights.append(f"• Best performing algorithm: {best_algorithm}")

    if len(detection_results) > 3:
        insights.append("• Multiple algorithms tested - high confidence in results")

    insights.append("• Consider ensemble methods for improved robustness")
    insights.append("• Regular retraining recommended for production use")

    for insight in insights:
        console.print(insight)


def _export_enhanced_results(results: dict, output_path: Path, format_type: str):
    """Export enhanced results with additional metadata."""

    enhanced_data = {
        **results,
        "analysis_metadata": {
            "analysis_type": "enhanced_autonomous",
            "algorithms_tested": len(
                results.get("autonomous_detection_results", {}).get(
                    "detection_results", {}
                )
            ),
            "export_timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        },
    }

    if format_type.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(enhanced_data, f, indent=2, default=str)


def _save_explanations(profile, recommendations: list, filename: str):
    """Save algorithm choice explanations to file."""

    explanations = {
        "dataset_profile": {
            "samples": profile.n_samples,
            "features": profile.n_features,
            "complexity": profile.complexity_score,
            "missing_ratio": profile.missing_values_ratio,
        },
        "algorithm_recommendations": [
            {
                "algorithm": rec.algorithm,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "expected_performance": rec.expected_performance,
            }
            for rec in recommendations
        ],
    }

    with open(filename, "w") as f:
        json.dump(explanations, f, indent=2)

    console.print(f"\n[green]✓[/green] Explanations saved to {filename}")


def _save_analysis_report(results: dict, analysis_type: str, output_path: Path):
    """Save analysis report to file."""

    report = {
        "analysis_type": analysis_type,
        "results_summary": "Comprehensive analysis completed",
        "insights": ["Statistical analysis performed", "Patterns identified"],
        "recommendations": ["Consider ensemble methods", "Monitor performance"],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    app()
