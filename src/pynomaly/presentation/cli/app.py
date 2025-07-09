"""CLI application using Typer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Check if we should use lazy loading (default: yes)
USE_LAZY_LOADING = os.getenv("PYNOMALY_USE_LAZY_CLI", "true").lower() == "true"

if USE_LAZY_LOADING:
    # Use lazy loading for better performance
    from pynomaly.presentation.cli.lazy_app import app
else:
    # Use traditional loading (for debugging/testing)
    from pynomaly.presentation.cli import automl, autonomous
    from pynomaly.presentation.cli import config as config_cli
    from pynomaly.presentation.cli import (
        datasets,
        deep_learning,
        detection,
        detectors,
        explainability,
        migrations,
        preprocessing,
        selection,
        server,
        tdd,
        validation,
    )
    from pynomaly.presentation.cli.export import export_app
    from pynomaly.presentation.cli.performance import performance_app

    # Configuration management CLI
    try:
        from pynomaly.presentation.cli import recommendation

        RECOMMENDATION_CLI_AVAILABLE = True
    except ImportError:
        recommendation = None
        RECOMMENDATION_CLI_AVAILABLE = False

    # Create Typer app
    app = typer.Typer(
        name="pynomaly",
        help="Pynomaly - State-of-the-art anomaly detection CLI",
        add_completion=True,
        rich_markup_mode="rich",
    )

    # Add subcommands
    app.add_typer(
        autonomous.app,
        name="auto",
        help="Autonomous anomaly detection (auto-configure and run)",
    )
    app.add_typer(
        automl.app, name="automl", help="Advanced AutoML & hyperparameter optimization"
    )
    app.add_typer(
        config_cli.app,
        name="config",
        help="Configuration management (capture, export, import)",
    )
    app.add_typer(detectors.app, name="detector", help="Manage anomaly detectors")
    app.add_typer(datasets.app, name="dataset", help="Manage datasets")
    app.add_typer(
        preprocessing.app,
        name="data",
        help="Data preprocessing (clean, transform, pipeline)",
    )
    app.add_typer(detection.app, name="detect", help="Run anomaly detection")
    app.add_typer(
        tdd.app,
        name="tdd",
        help="Test-Driven Development (TDD) management and enforcement",
    )
    app.add_typer(
        deep_learning.app,
        name="deep-learning",
        help="🧠 Deep learning anomaly detection (PyTorch, TensorFlow, JAX)",
    )
    app.add_typer(
        explainability.app,
        name="explainability",
        help="🔍 Explainable AI (model interpretability, bias analysis)",
    )
    app.add_typer(
        selection.app,
        name="selection",
        help="🧠 Intelligent algorithm selection with learning capabilities",
    )
    app.add_typer(
        export_app,
        name="export",
        help="Export results to business intelligence platforms",
    )
    app.add_typer(server.app, name="server", help="Manage API server")
    app.add_typer(
        performance_app, name="perf", help="Performance monitoring and optimization"
    )
    app.add_typer(
        validation.app,
        name="validate",
        help="🔍 Enhanced validation with rich output and GitHub integration",
    )
    app.add_typer(
        migrations.app,
        name="migrate",
        help="🗄️ Database migration management",
    )

    # Configuration recommendation commands
    if RECOMMENDATION_CLI_AVAILABLE:
        app.add_typer(
            recommendation.app,
            name="recommend",
            help="🧠 Intelligent configuration recommendations",
        )

from pynomaly.presentation.cli.container import get_cli_container

# Create console for rich output
console = Console()


@app.command()
def version():
    """Show version information."""
    container = get_cli_container()
    settings = container.config()

    console.print(f"[bold blue]Pynomaly[/bold blue] v{settings.app.version}")
    console.print(f"Python {sys.version.split()[0]}")
    console.print(f"Storage: {settings.storage_path}")


@app.command()
def settings(
    show: bool = typer.Option(False, "--show", help="Show current settings"),
    set_key: str | None = typer.Option(None, "--set", help="Set setting key=value"),
):
    """Manage application settings."""
    container = get_cli_container()
    settings = container.config()

    if show:
        # Show configuration
        table = Table(title="Pynomaly Settings")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        # Show key settings
        config_items = [
            ("App Name", settings.app_name),
            ("Version", settings.version),
            ("Debug Mode", str(settings.debug)),
            ("Storage Path", str(settings.storage_path)),
            ("API Host", settings.api_host),
            ("API Port", str(settings.api_port)),
            ("Max Dataset Size (MB)", str(settings.max_dataset_size_mb)),
            ("Default Contamination Rate", str(settings.default_contamination_rate)),
            ("GPU Enabled", str(settings.gpu_enabled)),
        ]

        for key, value in config_items:
            table.add_row(key, value)

        console.print(table)

    elif set_key:
        # Parse key=value
        if "=" not in set_key:
            console.print("[red]Error:[/red] Use format: --set key=value")
            raise typer.Exit(1)

        key, value = set_key.split("=", 1)

        # Update config (in real app, would persist this)
        console.print("[yellow]Note:[/yellow] Setting update not yet implemented")
        console.print(f"Would set: {key} = {value}")

    else:
        console.print("Use --show to display settings or --set key=value to update")


@app.command()
def status():
    """Show system status."""
    container = get_cli_container()

    # Get repository counts
    detector_count = container.detector_repository().count()
    dataset_count = container.dataset_repository().count()
    result_count = container.result_repository().count()

    # Create status table
    table = Table(title="Pynomaly System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Count", style="yellow")

    table.add_row("Detectors", "✓ Active", str(detector_count))
    table.add_row("Datasets", "✓ Active", str(dataset_count))
    table.add_row("Results", "✓ Active", str(result_count))
    table.add_row("API Server", "○ Not running", "-")

    console.print(table)

    # Show recent activity
    results = container.result_repository().find_recent(5)
    if results:
        console.print("\n[bold]Recent Detection Results:[/bold]")
        recent_table = Table()
        recent_table.add_column("Time", style="dim")
        recent_table.add_column("Detector", style="cyan")
        recent_table.add_column("Dataset", style="green")
        recent_table.add_column("Anomalies", style="red")

        for result in results:
            # Get detector and dataset names
            detector = container.detector_repository().find_by_id(result.detector_id)
            dataset = container.dataset_repository().find_by_id(result.dataset_id)

            recent_table.add_row(
                result.timestamp.strftime("%Y-%m-%d %H:%M"),
                detector.name if detector else "Unknown",
                dataset.name if dataset else "Unknown",
                f"{result.n_anomalies} ({result.anomaly_rate:.1%})",
            )

        console.print(recent_table)


@app.command()
def generate_config(
    config_type: str = typer.Argument(
        ..., help="Config type: 'test', 'experiment', or 'autonomous'"
    ),
    output: Path = typer.Option(
        "pynomaly_config.json", "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, yaml)"
    ),
    detector: str | None = typer.Option(None, "--detector", help="Detector algorithm"),
    dataset: str | None = typer.Option(None, "--dataset", help="Dataset path or name"),
    contamination: float | None = typer.Option(
        None, "--contamination", help="Contamination rate"
    ),
    max_algorithms: int | None = typer.Option(
        None, "--max-algorithms", help="Max algorithms to try"
    ),
    auto_tune: bool | None = typer.Option(
        None, "--auto-tune", help="Enable auto-tuning"
    ),
    cross_validation: bool | None = typer.Option(
        None, "--cv", help="Enable cross-validation"
    ),
    cv_folds: int | None = typer.Option(None, "--folds", help="Number of CV folds"),
    save_results: bool | None = typer.Option(None, "--save", help="Save results"),
    export_format: str | None = typer.Option(
        None, "--export-format", help="Export format"
    ),
    verbose: bool | None = typer.Option(None, "--verbose", help="Verbose output"),
    include_examples: bool = typer.Option(
        True, "--examples/--no-examples", help="Include usage examples"
    ),
):
    """Generate configuration files for tests or experiments from CLI options."""
    import json
    from datetime import datetime

    import yaml

    config = {
        "metadata": {
            "type": config_type,
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "description": f"Pynomaly {config_type} configuration",
        }
    }

    if config_type == "test":
        config["test"] = {
            "detector": {
                "algorithm": detector or "IsolationForest",
                "parameters": {
                    "contamination": contamination or 0.1,
                    "random_state": 42,
                },
            },
            "dataset": {
                "source": dataset or "path/to/test_data.csv",
                "validation": {
                    "enabled": True,
                    "checks": ["missing_values", "data_types", "duplicates"],
                },
            },
            "training": {
                "validate_data": True,
                "save_model": save_results if save_results is not None else True,
            },
            "detection": {
                "validate_features": True,
                "save_results": save_results if save_results is not None else True,
                "export": {
                    "enabled": True,
                    "format": export_format or "csv",
                    "path": "test_results.csv",
                },
            },
            "evaluation": {
                "cross_validation": (
                    cross_validation if cross_validation is not None else True
                ),
                "folds": cv_folds or 5,
                "metrics": ["precision", "recall", "f1", "auc_roc", "auc_pr"],
            },
        }

        if include_examples:
            config["examples"] = {
                "usage": [
                    "pynomaly detector create --name test_detector --algorithm IsolationForest",
                    "pynomaly dataset load path/to/test_data.csv --name test_data",
                    "pynomaly detect train test_detector test_data",
                    "pynomaly detect run test_detector test_data --output results.csv",
                    "pynomaly detect evaluate test_detector test_data --cv --folds 5",
                ],
                "description": "Basic testing workflow with single detector",
            }

    elif config_type == "experiment":
        config["experiment"] = {
            "name": "anomaly_detection_experiment",
            "description": "Comparative anomaly detection experiment",
            "algorithms": [
                {
                    "name": "IsolationForest",
                    "parameters": {
                        "contamination": contamination or 0.1,
                        "n_estimators": 100,
                    },
                },
                {
                    "name": "LOF",
                    "parameters": {
                        "contamination": contamination or 0.1,
                        "n_neighbors": 20,
                    },
                },
                {
                    "name": "OneClassSVM",
                    "parameters": {"nu": contamination or 0.1, "gamma": "scale"},
                },
            ],
            "dataset": {
                "source": dataset or "path/to/experiment_data.csv",
                "preprocessing": {
                    "normalization": "standard",
                    "feature_selection": False,
                    "outlier_removal": False,
                },
            },
            "evaluation": {
                "cross_validation": (
                    cross_validation if cross_validation is not None else True
                ),
                "folds": cv_folds or 5,
                "metrics": [
                    "precision",
                    "recall",
                    "f1",
                    "auc_roc",
                    "auc_pr",
                    "average_precision",
                ],
                "statistical_tests": ["wilcoxon", "friedman"],
            },
            "hyperparameter_tuning": {
                "enabled": auto_tune if auto_tune is not None else True,
                "method": "grid_search",
                "cv_folds": 3,
                "scoring": "average_precision",
            },
            "output": {
                "save_results": save_results if save_results is not None else True,
                "export_format": export_format or "excel",
                "include_visualizations": True,
                "generate_report": True,
            },
        }

        if include_examples:
            config["examples"] = {
                "usage": [
                    "pynomaly dataset load path/to/experiment_data.csv --name exp_data",
                    "pynomaly detect batch IsolationForest LOF OneClassSVM exp_data",
                    "pynomaly detect evaluate IsolationForest exp_data --cv --folds 5",
                    "pynomaly export excel results.json experiment_results.xlsx",
                ],
                "description": "Multi-algorithm experiment with statistical comparison",
            }

    elif config_type == "autonomous":
        config["autonomous"] = {
            "data_source": dataset or "path/to/data.csv",
            "analysis": {
                "max_samples": 10000,
                "profile_data": True,
                "detect_seasonality": True,
                "complexity_analysis": True,
            },
            "detection": {
                "max_algorithms": max_algorithms or 5,
                "confidence_threshold": 0.8,
                "auto_tune_hyperparams": auto_tune if auto_tune is not None else True,
                "ensemble_methods": True,
            },
            "output": {
                "save_results": save_results if save_results is not None else True,
                "export_results": True,
                "export_format": export_format or "csv",
                "verbose": verbose if verbose is not None else False,
                "generate_insights": True,
            },
            "quality_assurance": {
                "validation_checks": True,
                "result_consistency": True,
                "performance_monitoring": True,
            },
        }

        if include_examples:
            config["examples"] = {
                "usage": [
                    "pynomaly auto detect path/to/data.csv --output results.csv",
                    "pynomaly auto profile path/to/data.csv --verbose",
                    "pynomaly auto quick path/to/data.csv --contamination 0.05",
                ],
                "description": "Fully autonomous anomaly detection with minimal configuration",
            }

    else:
        console.print(
            f"[red]Error:[/red] Unknown config type '{config_type}'. Use 'test', 'experiment', or 'autonomous'"
        )
        raise typer.Exit(1)

    # Save configuration
    try:
        if format.lower() == "yaml":
            with open(output, "w") as f:
                yaml.dump(
                    config, f, default_flow_style=False, sort_keys=False, indent=2
                )
        else:  # json
            with open(output, "w") as f:
                json.dump(config, f, indent=2, default=str)

        console.print(
            f"[green]✓[/green] {config_type.title()} configuration generated: {output}"
        )

        if include_examples:
            console.print("\n[bold blue]Usage Examples:[/bold blue]")
            examples = config.get("examples", {}).get("usage", [])
            for example in examples:
                console.print(f"  [cyan]{example}[/cyan]")

            description = config.get("examples", {}).get("description", "")
            if description:
                console.print(f"\n[dim]{description}[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to save config: {str(e)}")
        raise typer.Exit(1)


@app.command()
def quickstart():
    """Run interactive quickstart guide."""
    console.print("[bold blue]Welcome to Pynomaly![/bold blue]\n")
    console.print("This quickstart will help you get started with anomaly detection.\n")

    # Check if user wants to continue
    if not typer.confirm("Would you like to continue?"):
        console.print("Quickstart cancelled.")
        raise typer.Exit()

    console.print("\n[bold]Step 1: Load a dataset[/bold]")
    console.print("You can load data from CSV or Parquet files.")
    console.print("Example: [cyan]pynomaly dataset load data.csv --name my_data[/cyan]")

    console.print("\n[bold]Step 2: Clean and preprocess data (optional)[/bold]")
    console.print("Clean missing values, outliers, and transform features.")
    console.print("Examples:")
    console.print(
        "  [cyan]pynomaly data clean <dataset_id> --missing drop_rows --outliers clip[/cyan]"
    )
    console.print(
        "  [cyan]pynomaly data transform <dataset_id> --scaling standard --encoding onehot[/cyan]"
    )

    console.print("\n[bold]Step 3: Create a detector[/bold]")
    console.print("Choose from various algorithms like IsolationForest, LOF, etc.")
    console.print(
        "Example: [cyan]pynomaly detector create my_detector --algorithm IsolationForest[/cyan]"
    )

    console.print("\n[bold]Step 4: Train the detector[/bold]")
    console.print("Train your detector on the loaded dataset.")
    console.print("Example: [cyan]pynomaly detect train my_detector my_data[/cyan]")

    console.print("\n[bold]Step 5: Detect anomalies[/bold]")
    console.print("Run detection on new data.")
    console.print("Example: [cyan]pynomaly detect run my_detector test_data[/cyan]")

    console.print("\n[bold]Step 6: View and export results[/bold]")
    console.print("Analyze detection results and export them to various platforms.")
    console.print("Examples:")
    console.print("  [cyan]pynomaly detect results --latest[/cyan]")
    console.print("  [cyan]pynomaly export excel results.json output.xlsx[/cyan]")
    console.print("  [cyan]pynomaly export list-formats[/cyan]")

    console.print(
        "\n[green]Ready to start![/green] Use [cyan]pynomaly --help[/cyan] to see all commands."
    )


@app.command()
def setup():
    """Interactive setup wizard for new users.

    This command provides a guided setup experience for users who are
    new to Pynomaly, helping them configure their first detection workflow.
    """
    try:
        from pynomaly.presentation.cli.ux_improvements import create_setup_wizard

        config = create_setup_wizard()

        if config:
            console.print("\n[cyan]Commands to run:[/cyan]")
            console.print(
                f"  [white]pynomaly dataset load {config['data_path']} --name my-dataset[/white]"
            )
            console.print(
                f"  [white]pynomaly detector create my-detector --algorithm {config['algorithm']} --contamination {config['contamination']}[/white]"
            )
            console.print(
                "  [white]pynomaly detect train my-detector my-dataset[/white]"
            )
            console.print(
                f"  [white]pynomaly detect run my-detector my-dataset --output results.{config['output_format']}[/white]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] Setup wizard failed: {str(e)}")
        raise typer.Exit(1)


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
):
    """Pynomaly - State-of-the-art anomaly detection platform."""
    if verbose and quiet:
        console.print("[red]Error:[/red] Cannot use --verbose and --quiet together")
        raise typer.Exit(1)

    # Set output level
    if quiet:
        console.quiet = True
    elif verbose:
        # In verbose mode, could set logging level
        pass


if __name__ == "__main__":
    app()
