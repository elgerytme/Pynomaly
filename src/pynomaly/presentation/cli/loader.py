"""Dynamic CLI loader that supports experimental features."""

import os
import sys
from typing import Optional

import typer
from rich.console import Console

from pynomaly.presentation.cli import (
    automl,
    autonomous,
    config as config_cli,
    datasets,
    dashboard,
    detection,
    detectors,
    governance,
    preprocessing,
    security,
    server,
    tdd,
)
from pynomaly.presentation.cli.container import get_cli_container
from pynomaly.presentation.cli.export import export_app
from pynomaly.presentation.cli.performance import performance_app


def create_app() -> typer.Typer:
    """Create the CLI app with dynamic feature loading."""
    # Check for experimental flag early
    experimental = "--experimental" in sys.argv
    
    # Set environment variables for experimental features
    if experimental:
        os.environ["PYNOMALY_EXPERIMENTAL"] = "true"
        os.environ["PYNOMALY_DEEP_LEARNING"] = "true"
        os.environ["PYNOMALY_ADVANCED_EXPLAINABILITY"] = "true"
        os.environ["PYNOMALY_INTELLIGENT_SELECTION"] = "true"
    
    # Create the main app
    app = typer.Typer(
        name="pynomaly",
        help="Pynomaly - State-of-the-art anomaly detection CLI",
        add_completion=True,
        rich_markup_mode="rich",
    )
    
    console = Console()
    
    # Add core commands
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
        tdd.app, name="tdd", help="Test-Driven Development (TDD) management and enforcement"
    )
    app.add_typer(security.app, name="security", help="🔒 Security & compliance (SOC2, GDPR, HIPAA, encryption)")
    app.add_typer(dashboard.app, name="dashboard", help="📊 Advanced visualization dashboards (executive, operational, analytical)")
    app.add_typer(
        export_app, name="export", help="Export results to business intelligence platforms"
    )
    app.add_typer(server.app, name="server", help="Manage API server")
    app.add_typer(governance.app, name="governance", help="Governance framework and audit management commands.")
    app.add_typer(
        performance_app, name="perf", help="Performance monitoring and optimization"
    )
    
    # Add deep learning commands (now enabled by default for P-003)
    try:
        from pynomaly.presentation.cli import deep_learning
        app.add_typer(deep_learning.app, name="deep-learning", help="🧠 Deep learning anomaly detection (PyTorch, TensorFlow, JAX)")
    except ImportError as e:
        console.print(f"[yellow]Warning: Could not load deep learning features: {e}[/yellow]")
    
    # Add experimental commands if enabled
    if experimental or os.environ.get("PYNOMALY_EXPERIMENTAL", "false").lower() == "true":
        try:
            from pynomaly.presentation.cli import explainability, selection
            app.add_typer(explainability.app, name="explainability", help="🔍 Explainable AI (model interpretability, bias analysis)")
            app.add_typer(selection.app, name="selection", help="🧠 Intelligent algorithm selection with learning capabilities")
        except ImportError as e:
            console.print(f"[yellow]Warning: Could not load experimental features: {e}[/yellow]")
    
    @app.command()
    def version():
        """Show version information."""
        container = get_cli_container()
        settings = container.config()
        console.print(f"[bold blue]Pynomaly[/bold blue] v{settings.app.version}")
        console.print(f"Python {sys.version.split()[0]}")
        console.print(f"Storage: {settings.storage_path}")
    
    @app.callback()
    def main(
        verbose: bool = typer.Option(
            False, "--verbose", "-v", help="Enable verbose output"
        ),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
        experimental: bool = typer.Option(
            False, "--experimental", help="Enable experimental features"
        ),
    ):
        """Pynomaly - State-of-the-art anomaly detection platform."""
        if verbose and quiet:
            console.print("[red]Error:[/red] Cannot use --verbose and --quiet together")
            raise typer.Exit(1)
        
        if experimental:
            console.print("[yellow]⚠️ Experimental features enabled[/yellow]")
        
        # Set output level
        if quiet:
            console.quiet = True
        elif verbose:
            # In verbose mode, could set logging level
            pass
    
    return app
